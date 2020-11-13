from client import Client
import numpy as np
import pickle


if __name__ == "__main__":
    
    client = Client("0.0.0.0", 5002)

    client.send("WP2")

    print("conneceted")
    
    # answer = client.receive()
    # print(answer)

    B = np.arange(0, 4*4, dtype=float).reshape((4,4))

    B = np.array([ 
        [0.2510353932677801, 0.967252269086237, 0.03747371444648634, -0.14494078541806812],
        [-0.8258221556266007, 0.19381218223186758, 0.5295796496229401, 0.09035400829947654],
        [0.5049742553864758, -0.16388985925324095, 0.8474320712781805, -0.3412307288854375],
        [0.0,0.0,0.0,1.0]
    ])

    count = 0
    
    while(True):
        ## Serialize pose matrix and send
        serializedPose = pickle.dumps(B)
        client.send(serializedPose, encode=False)

        ## Wait for confirmation
        reply = client.receive()
        count = count + 1
        print(count, "[Server]: {}".format(reply))

        if("ACK" in reply):
            continue

        print("[Client]: No ACK received. Exiting...")
        break
