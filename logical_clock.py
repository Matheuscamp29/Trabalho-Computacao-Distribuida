from threading import Lock

valor = 0
trava = Lock()


def tick():
    global valor
    with trava:
        valor += 1
        return valor


def receive(remoto):
    global valor
    if remoto < 0:
        raise ValueError("timestamp de Lamport deve ser não negativo")
    with trava:
        valor = max(valor, remoto) + 1
        return valor


def atual():
    with trava:
        return valor
