# point to point request-reply stream
from typing import Union
import dataclasses
import logging
import pickle
import socket
import time

import torch
import zmq

import api.config
import base.name_resolve as name_resolve
import base.namedarray as namedarray
import base.names as names

logger = logging.getLogger("Request-Replay Stream")


class RequestClient:
    # in master server
    def post_request(self, payload):
        raise NotImplementedError

    def poll_reply(self):
        raise NotImplementedError


class ReplyServer:
    # in model worker
    def poll_request(self):
        raise NotImplementedError

    def post_reply(self):
        raise NotImplementedError


@dataclasses.dataclass
class Request:
    handle_name: str  # handle names
    data: namedarray.NamedArray = None


@dataclasses.dataclass
class Reply:
    data: namedarray.NamedArray = None


class IpRequestClient(RequestClient):

    def __init__(self, address, serialization_method):
        self.__context = zmq.Context()
        self.__socket = self.__context.socket(zmq.REQ)
        self.__socket.connect(f"tcp://{address}")
        self.__socket.setsockopt(zmq.LINGER, 0)

        self.__serialization_method = serialization_method

    def post_request(self, payload: Request):
        tik = time.perf_counter()
        if isinstance(payload.data, namedarray.NamedArray):
            assert isinstance(payload.data, namedarray.NamedArray), type(payload.data)
            payload.data = namedarray.recursive_apply(payload.data, lambda x: x.cpu().numpy())
            payload.data = namedarray.dumps(payload.data, method=self.__serialization_method)
            encoding = b'01'
        else:
            payload.data = [pickle.dumps(payload.data)]
            encoding = b'00'
        logger.debug(f"Request encoding time: {time.perf_counter() - tik:.4f}s")
        self.__socket.send_multipart([payload.handle_name.encode('ascii'), encoding] + payload.data)

    def poll_reply(self):
        encoding, *data = self.__socket.recv_multipart()
        tik = time.perf_counter()
        if encoding == b'01':
            data = namedarray.loads(data)
            data = namedarray.recursive_apply(data, lambda x: torch.from_numpy(x))
        elif encoding == b'00':
            data = pickle.loads(data[0])
        else:
            raise NotImplementedError()
        logger.debug(f"Reply decoding time: {time.perf_counter() - tik:.4f}s")
        return Reply(data)


class IpReplyServer(ReplyServer):

    def __init__(self, serialization_method):  # auto find port
        self.__context = zmq.Context()
        self.__socket = self.__context.socket(zmq.REP)
        host_ip = socket.gethostbyname(socket.gethostname())
        port = self.__socket.bind_to_random_port(f"tcp://{host_ip}")
        self.address = f"{host_ip}:{port}"
        self.__socket.setsockopt(zmq.LINGER, 0)
        self.__serialization_method = serialization_method

    def poll_request(self):
        handle_name, encoding, *data = self.__socket.recv_multipart()
        handle_name = handle_name.decode('ascii')
        tik = time.perf_counter()
        if encoding == b'01':
            data = namedarray.loads(data)
            data = namedarray.recursive_apply(data, lambda x: torch.from_numpy(x))
        elif encoding == b'00':
            data = pickle.loads(data[0])
        else:
            raise NotImplementedError()
        logger.debug(f"Request decoding time: {time.perf_counter() - tik:.4f}s")
        return Request(handle_name, data)

    def post_reply(self, payload: Reply):
        tik = time.perf_counter()
        if isinstance(payload.data, namedarray.NamedArray):
            assert isinstance(payload.data, namedarray.NamedArray), type(payload.data)
            payload.data = namedarray.recursive_apply(payload.data, lambda x: x.cpu().numpy())
            payload.data = namedarray.dumps(payload.data, method=self.__serialization_method)
            encoding = b'01'
        else:
            payload.data = [pickle.dumps(payload.data)]
            encoding = b'00'
        logger.debug(f"Reply encoding time: {time.perf_counter() - tik:.4f}s")
        self.__socket.send_multipart([encoding] + payload.data)


class NameResolvingRequestClient(IpRequestClient):

    def __init__(self, experiment_name, trial_name, stream_name,
                 serialization_method):  # name should be formatted as {from_worker_name}_{to_worker_name}
        # post address
        name = names.request_reply_stream(experiment_name, trial_name, stream_name)
        address = name_resolve.wait(name, timeout=15)
        super().__init__(address, serialization_method)


class NameResolvingReplyServer(IpReplyServer):

    def __init__(self, experiment_name, trial_name, stream_name, serialization_method):
        super().__init__(serialization_method)
        name = names.request_reply_stream(experiment_name, trial_name, stream_name)
        name_resolve.add(name=name, value=self.address)


def make_request_client(worker_info: api.config.WorkerInformation,
                        config: Union[str, api.config.RequestReplyStream]):
    if isinstance(config, str):
        config = api.config.RequestReplyStream(stream_name=config)
    return NameResolvingRequestClient(worker_info.experiment_name, worker_info.trial_name, config.stream_name,
                                      config.serialization_method)


def make_reply_server(worker_info: api.config.WorkerInformation,
                      config: Union[str, api.config.RequestReplyStream]):
    if isinstance(config, str):
        config = api.config.RequestReplyStream(stream_name=config)
    return NameResolvingReplyServer(worker_info.experiment_name, worker_info.trial_name, config.stream_name,
                                    config.serialization_method)