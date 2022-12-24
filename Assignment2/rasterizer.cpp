// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

// Determine whether two vectors v1 and v2 point to the same direction
// v1 = Cross(AB, AC)
// v2 = Cross(AB, AP)
static bool SameSide(Vector3f A, Vector3f B, Vector3f C, Vector3f P)
{
    Vector3f AB = B - A;
    Vector3f AC = C - A;
    Vector3f AP = P - A;

    Vector3f v1 = AB.cross(AC); 
    Vector3f v2 = AB.cross(AP);

    // v1 and v2 should point to the same direction
    return v1.dot(v2) >= 0;
}

// https://www.cnblogs.com/graphics/archive/2010/08/05/1793393.html
static bool insideTriangle(float x, float y, const Vector3f* _v)
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    Vector3f A(_v[0].x(), _v[0].y(), 0.0);
    Vector3f B(_v[1].x(), _v[1].y(), 0.0);
    Vector3f C(_v[2].x(), _v[2].y(), 0.0);

    Vector3f P(x, y, 0.0);

    return SameSide(A, B, C, P) &&
        SameSide(B, C, A, P) &&
        SameSide(C, A, B, P);
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4(); //把顶点变成四维向量
    
    // TODO : Find out the bounding box of current triangle.
    // iterate through the pixel and find if the current pixel is inside the triangle
    std::array<Vector2f, 3> aabb = t.getAABBBox();
    int min_x = std::floor(aabb[0][0]);
    int max_x = std::ceil(aabb[0][1]);
    int min_y = std::floor(aabb[1][0]);
    int max_y = std::ceil(aabb[1][1]);

    //超采样
    std::array<Vector2f, 4> offsetArr;
    offsetArr[0] = Vector2f(0.25, 0.25);
    offsetArr[1] = Vector2f(0.25, 0.75);
    offsetArr[2] = Vector2f(0.75, 0.25);
    offsetArr[3] = Vector2f(0.75, 0.75);

    for (int x = min_x; x < max_x; ++x)
    {
        for (int y = min_y; y < max_y; ++y)
        {
            //if (insideTriangle(x, y, t.v))
            //{
            //    //https://blog.csdn.net/Q_pril/article/details/123598746
            //    //-------------------
            //    auto [alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
            //    float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
            //    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
            //    z_interpolated *= w_reciprocal;
            //    //深度插值与误差修复

            //    float current_z = depth_buf[get_index(x, y)];
            //    if (z_interpolated < current_z)
            //    {
            //        depth_buf[get_index(x, y)] = z_interpolated;
            //        set_pixel(Vector3f(x, y, 0), t.getColor());
            //    }
            //   
            //}
            bool superDepthTest = false;
            for (int idx = 0; idx < offsetArr.size(); idx++)
            {
                float newX = x + offsetArr[idx][0];
                float newY = y + offsetArr[idx][1];

                if (insideTriangle(newX, newY, t.v))
                {
                    //https://blog.csdn.net/Q_pril/article/details/123598746
                    //-------------------
                    auto [alpha, beta, gamma] = computeBarycentric2D(newX, newY, t.v);
                    float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;
                    //深度插值与误差修复

                    float current_z = superdepth_buf[get_index(x, y) * 4 + idx];
                    if (z_interpolated < current_z)
                    {
                        superdepth_buf[get_index(x, y) * 4 + idx] = z_interpolated;
                        superframe_buf[get_index(x, y) * 4 + idx] = t.getColor();
                        superDepthTest = true;
                        //set_pixel(Vector3f(x, y, 0), t.getColor(), idx);
                    }

                }
            }
            if (superDepthTest)
            {
                set_pixel(Vector3f(x, y, 0), (superframe_buf[get_index(x, y) * 4 + 0] 
                                                + superframe_buf[get_index(x, y) * 4 + 1]
                                                + superframe_buf[get_index(x, y) * 4 + 2]
                                                + superframe_buf[get_index(x, y) * 4 + 3]) / 4.0);
            }
            
        }
    }


    // If so, use the following code to get the interpolated z value.
    /*auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
    float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    z_interpolated *= w_reciprocal;*/

    // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
        std::fill(superframe_buf.begin(), superframe_buf.end(), Eigen::Vector3f{ 0, 0, 0 });
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        float infty = std::numeric_limits<float>::infinity();
        std::fill(depth_buf.begin(), depth_buf.end(), infty);
        std::fill(superdepth_buf.begin(), superdepth_buf.end(), infty);
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
    superdepth_buf.resize(w * h * 4);
    superframe_buf.resize(w * h * 4);
}

int rst::rasterizer::get_index(int x, int y, int offset)
{
    //FIXMEjhh 这里需要加保护 防止越界 
    return (height-1-y)*width + x + offset;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color, int offset)
{
    //FIXMEjhh 这里需要加保护 防止越界 
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x() + offset;
    frame_buf[ind] = color;

}

// clang-format on