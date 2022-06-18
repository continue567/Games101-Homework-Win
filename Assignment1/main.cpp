#include "Triangle.hpp"
#include "rasterizer.hpp"
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1,
        -eye_pos[2], 0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the Z axis.
    // Then return it.
    
    float angle_pi = rotation_angle * MY_PI / 180.0;
    model << cos(angle_pi), -sin(angle_pi), 0, 0,
             sin(angle_pi), cos(angle_pi), 0, 0,
             0, 0, 1, 0, 
             0, 0, 0, 1;


    return model;
}

Eigen::Matrix4f get_model_matrix(Vector3f axis, float angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    float angle_pi = -angle * MY_PI / 180.0;
    model << std::pow(axis.x(), 2)*(1.0 - cos(angle_pi))+ cos(angle_pi),             axis.x() * axis.y() * (1.0 - cos(angle_pi)) + axis.z() * sin(angle_pi),    axis.x()* axis.z()* (1.0 - cos(angle_pi)) - axis.y() * sin(angle_pi),           0,
             axis.x()* axis.y()* (1.0 - cos(angle_pi)) - axis.z() * sin(angle_pi),   std::pow(axis.y(), 2)* (1.0 - cos(angle_pi)) + cos(angle_pi),              axis.y()* axis.z()* (1.0 - cos(angle_pi)) + axis.x() * sin(angle_pi),           0,
             axis.x()* axis.z()* (1.0 - cos(angle_pi)) + axis.y() * sin(angle_pi),   axis.y()* axis.z()* (1.0 - cos(angle_pi)) - axis.x() * sin(angle_pi),      std::pow(axis.z(), 2)* (1.0 - cos(angle_pi)) + cos(angle_pi),                   0,
             0,                                                                      0,                                                                         0,                                                                              1;
    return model;
}


Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    // Students will implement this function
    float angle_pi = eye_fov * MY_PI / 180.0;
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the projection matrix for the given parameters.
    // Then return it.
    projection <<   1.0/(tan(angle_pi)*aspect_ratio), 0, 0, 0,
                    0, 1.0 / tan(angle_pi), 0, 0,
                    0, 0, -(zFar + zNear)/(zFar - zNear), -2.0* zFar * zNear / (zFar - zNear),
                    0, 0, -1, 0;

    return projection;
}

int main(int argc, const char** argv)
{
    constexpr int screen_width = 500;
    constexpr int screen_height = 500;
    float angle = 0.0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3) {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {
            filename = std::string(argv[3]);
        }
        else
            return 0;
    }

    rst::rasterizer r(screen_width, screen_height);

    Eigen::Vector3f eye_pos = {0, 0, 10};

    std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};

    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;
    int frame_count = 0;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(screen_width, screen_height, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(Eigen::Vector3f(0.0, 1.0, 0.0),  angle));
        //r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(screen_width, screen_height, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a') {
            angle += 1;
        }
        else if (key == 'd') {
            angle -= 1;
        }
    }

    return 0;
}
