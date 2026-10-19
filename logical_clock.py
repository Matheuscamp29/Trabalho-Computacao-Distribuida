"""Relógio lógico de Lamport compartilhado por cada processo."""

from threading import Lock


class LamportClock:
    def __init__(self):
        self._value = 0
        self._lock = Lock()

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

    def tick(self) -> int:
        """Registra um evento local, incluindo o envio de uma mensagem."""
        with self._lock:
            self._value += 1
            return self._value

    def receive(self, remote_timestamp: int) -> int:
        """Registra o recebimento: C = max(C, timestamp recebido) + 1."""
        if remote_timestamp < 0:
            raise ValueError("timestamp de Lamport deve ser não negativo")
        with self._lock:
            self._value = max(self._value, remote_timestamp) + 1
            return self._value
