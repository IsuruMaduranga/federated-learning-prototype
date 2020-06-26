import socket
import pickle
import select
import sys
import numpy as np
import pandas as pd

class Client:

    def __init__(self, MODEL, weights_path, HEADER_LENGTH = 10, IP = socket.gethostname(), PORT = 5000):

        self.HEADER_LENGTH =  HEADER_LENGTH
        self.IP = IP
        self.PORT = PORT

        self.weights_path = weights_path

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((IP, PORT))

        self.MODEL = MODEL
        self.STOP_FLAG = False
        self.rounds = 0


    def send_model(self):
        data = np.array(self.MODEL.get_weights())
        data = pickle.dumps(data)
        data = bytes(f"{len(data):<{self.HEADER_LENGTH}}", 'utf-8') + data
        self.client_socket.sendall(data)


    def receive(self):
        try:

            message_header = self.client_socket.recv(self.HEADER_LENGTH)
            if not len(message_header):
                return False

            message_length = int(message_header.decode('utf-8').strip())

            full_msg = b''
            while True:
                msg = self.client_socket.recv(message_length)

                full_msg += msg

                if len(full_msg) == message_length:
                    break
            
            data = pickle.loads(full_msg)

            self.STOP_FLAG = data["STOP_FLAG"]

            return data["WEIGHTS"]

        except Exception as e:
            print(e)


    def fetch_model(self):
        data = self.receive()

        self.MODEL.set_weights(data)

        np.save(self.weights_path,data)

    def train(self):
        self.MODEL.fit()

    def run(self):

        while not self.STOP_FLAG:

            read_sockets, _, exception_sockets = select.select([self.client_socket], [], [self.client_socket])

            for soc in read_sockets:
                self.fetch_model()
            
            if self.STOP_FLAG:
                break

            print(f"Model version: {self.rounds} fetched")

            self.rounds += 1
            print(f"Training cycle: {self.rounds}")
            self.train()

            print(f"Sent local model")
            self.send_model()

        print("Training Done")


if __name__ == "__main__":

    from models.model import Model

    path_weights = sys.argv[1]
    path_node_partition = sys.argv[2]
    path_edge_partition = sys.argv[3]

    nodes = pd.read_csv(path_node_partition , sep='\t', lineterminator='\n',header=None).loc[:,0:1433]
    nodes.set_index(0,inplace=True)

    edges = pd.read_csv(path_edge_partition , sep='\s+', lineterminator='\n', header=None)
    edges.columns = ["source","target"] 

    model = Model(nodes,edges)
    model.initialize()

    client = Client(model,weights_path=path_weights)

    client.run()

    
