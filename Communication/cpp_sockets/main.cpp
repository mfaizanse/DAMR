#include <iostream>
#include "client.hpp"
#include <unistd.h>

std::string convertMatToString(cv::Mat mat) {
    std::string str = "";
    for (unsigned int i = 0; i < mat.rows; i++) {
        for (unsigned int j = 0; j < mat.cols; j++) {
            if (str != "") {
                str = str + ",";
            }
            str = str + std::to_string(mat.at<float>(i,j));
        }
    }

    return str;
}

int main() {
    std::cout << "MAIN START" << std::endl;

    // Load image
    cv::Mat img = cv::imread("../lena.png");

    socket_communication::Client client("127.0.0.1", 5002);
    // Check that connection works
    client.Send("WP2");
    std::string answer = client.Receive();
    std::cout << "[Server]: " << answer << std::endl;

    cv::Mat B(4,4,CV_32FC1);
    for (unsigned int i = 0; i < B.rows; i++) {
        for (unsigned int j = 0; j < B.cols; j++) {
            B.at<float>(i,j) = ((float)(i * B.cols + j)) / 10;
        }
    }

    while (true) {
        std::string poseStr = convertMatToString(B);

//        std::cout << "[Client (Pose)]: " << poseStr << std::endl;
        client.Send(poseStr);

        std::string reply = client.Receive();
        //std::cout << "[Server]: " << reply << std::endl;

        // wait for server to send ACk back before sending new pose
        if (reply.find("ACK") != std::string::npos) {
            // send next pose
            continue;
        }

        std::cout << "Client: Not received ACK. Exiting..." << std::endl;
        break;

//        client.SendImage(img);
//        std::string msg = client.Receive();
//        std::cout << "[Server]: " << msg << std::endl << "Client: ";
//        int a;
//        std::cin >> a;
//        std::cout << std::endl;
//        break;
    }

}