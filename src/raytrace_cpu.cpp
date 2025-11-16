// Compile: g++ -O3 -std=c++17 raytrace_spheres_improved_sequential.cpp -o raytrace_sequential
// Run: ./raytrace_sequential

#include <iomanip>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

struct Vec3 {
    float x,y,z;
    inline Vec3(): x(0),y(0),z(0) {}
    inline Vec3(float a,float b,float c): x(a),y(b),z(c) {}
    inline Vec3 operator+(const Vec3 &o) const { return Vec3(x+o.x,y+o.y,z+o.z); }
    inline Vec3 operator-(const Vec3 &o) const { return Vec3(x-o.x,y-o.y,z-o.z); }
    inline Vec3 operator*(float s) const { return Vec3(x*s,y*s,z*s); }
    inline Vec3 operator*(const Vec3 &o) const { return Vec3(x*o.x,y*o.y,z*o.z); }
    inline Vec3 operator/(float s) const { float inv = 1.0f/s; return Vec3(x*inv,y*inv,z*inv); }
    inline float length() const { return std::sqrt(x*x + y*y + z*z); }
    inline Vec3 normalized() const { float l = length(); return l>0 ? (*this)/l : Vec3(0,0,0); }
};

inline float dot(const Vec3 &a,const Vec3 &b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Vec3 cross(const Vec3 &a,const Vec3 &b){
    return Vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
inline Vec3 reflect(const Vec3 &I,const Vec3 &N){ return I - N*(2.0f*dot(I,N)); }
inline float clampf(float v,float a,float b){ return std::clamp(v, a, b); }

struct Ray { Vec3 o,d; inline Ray(){} inline Ray(const Vec3 &o_, const Vec3 &d_): o(o_), d(d_.normalized()) {} };

struct Material {
    Vec3 albedo;
    float reflectivity;
    float specular_exponent;
    float absorption;
    inline Material(): albedo(1,1,1), reflectivity(0), specular_exponent(8), absorption(0) {}
    inline Material(const Vec3 &a,float r,float s,float abs): albedo(a), reflectivity(r), specular_exponent(s), absorption(abs) {}
};

struct Sphere {
    Vec3 center; float radius; Material mat;
    inline Sphere() {}
    inline Sphere(const Vec3 &c,float r,const Material &m): center(c), radius(r), mat(m) {}
    bool intersect(const Ray &ray, float &tHit, Vec3 &n) const {
        Vec3 oc = ray.o - center;
        float a = dot(ray.d, ray.d);
        float b = 2.0f * dot(oc, ray.d);
        float c = dot(oc, oc) - radius*radius;
        float disc = b*b - 4*a*c;
        if (disc < 0) return false;
        float sq = std::sqrt(disc);
        float t0 = (-b - sq) / (2*a);
        float t1 = (-b + sq) / (2*a);
        float t = t0;
        if (t < 1e-4f) t = t1;
        if (t < 1e-4f) return false;
        tHit = t;
        Vec3 p = ray.o + ray.d * t;
        n = (p - center).normalized();
        return true;
    }
};

Vec3 background_color(const Vec3 &dir) {
    float t = 0.5f * (dir.y + 1.0f);
    Vec3 c1(0.7f,0.9f,1.0f), c2(0.15f,0.16f,0.2f);
    return c1 * t + c2 * (1.0f - t);
}

Vec3 trace_ray(Ray ray, const std::vector<Sphere> &spheres,
               const Vec3 &camPos, const Vec3 &lightPos, const Vec3 &lightColor, int max_depth) {

    Vec3 throughput(1,1,1);
    Vec3 radiance(0,0,0);
    int nspheres = spheres.size();

    for (int depth = 0; depth < max_depth; ++depth) {
        float bestT = 1e20f;
        int hitIdx = -1;
        Vec3 N;
        for (int i=0;i<nspheres;++i){
            float tHit; Vec3 n;
            if (spheres[i].intersect(ray, tHit, n)){
                if (tHit < bestT){ bestT = tHit; hitIdx = i; N = n; }
            }
        }
        if (hitIdx == -1) {
            radiance = radiance + throughput * background_color(ray.d);
            break;
        }

        Vec3 hitP = ray.o + ray.d * bestT;
        Material mat = spheres[hitIdx].mat;

        if (mat.absorption > 0.0f)
            throughput = throughput * std::exp(-mat.absorption * bestT);

        Vec3 toLight = (lightPos - hitP);
        float distToLight = toLight.length();
        Vec3 L = toLight / distToLight;
        Ray shadowRay(hitP + N * 1e-4f, L);
        bool inShadow = false;
        for (int i=0;i<nspheres;++i){
            float tShadow; Vec3 tmpN;
            if (spheres[i].intersect(shadowRay, tShadow, tmpN))
                if (tShadow < distToLight - 1e-4f) { inShadow = true; break; }
        }

        float NdotL = std::max(dot(N, L), 0.0f);
        Vec3 diffuse = mat.albedo * NdotL;

        Vec3 V = (camPos - hitP).normalized();
        Vec3 H = (V + L).normalized();
        float NdotH = std::max(dot(N, H), 0.0f);
        float spec = std::pow(NdotH, mat.specular_exponent);
        Vec3 specular = lightColor * spec * mat.reflectivity;

        Vec3 direct = (!inShadow) ? (diffuse * lightColor + specular) : Vec3(0,0,0);
        Vec3 ambient = mat.albedo * 0.05f;
        float nonRef = 1.0f - mat.reflectivity;
        radiance = radiance + throughput * ((direct + ambient) * nonRef);

        if (mat.reflectivity > 0.0f) {
            Vec3 R = reflect(ray.d, N).normalized();
            ray.o = hitP + R * 1e-4f;
            ray.d = R;
            throughput = throughput * mat.albedo * mat.reflectivity;
            if (throughput.x < 1e-4f && throughput.y < 1e-4f && throughput.z < 1e-4f) break;
        } else break;
    }
    return radiance;
}

int main() {
    const int width = 1920;
    const int height = 1080;
    const int numPixels = width * height;
    const int samples = 10;
    const int max_depth = 6;

    std::vector<Sphere> spheres;

    spheres.emplace_back(Vec3(0.0f, -1005.0f, -20.0f), 1000.0f, Material(Vec3(0.8f,0.8f,0.8f), 0.0f, 16.0f, 0.01f));
    spheres.emplace_back(Vec3(-2.5f, 0.5f, -10.0f), 2.0f, Material(Vec3(0.95f,0.95f,0.98f), 0.95f, 64.0f, 0.02f));
    spheres.emplace_back(Vec3(0.0f, 1.2f, -6.0f), 1.2f, Material(Vec3(0.2f,0.9f,0.35f), 0.6f, 32.0f, 0.08f));
    spheres.emplace_back(Vec3(3.5f, 0.3f, -6.0f), 1.0f, Material(Vec3(0.6f,0.7f,0.95f), 0.7f, 100.0f, 0.03f));
    spheres.emplace_back(Vec3(1.0f, 0.0f, -4.0f), 0.8f, Material(Vec3(0.9f, 0.1f, 0.1f), 0.0f, 16.0f, 0.0f));
    //spheres.emplace_back(Vec3(3.8f, 1.0f, -3.0f), 1.0f, Material(Vec3(0.6f,0.7f,0.95f), 0.7f, 64.0f, 0.03f));

    Vec3 camPos(0.0f, 1.0f, 2.5f);
    Vec3 camLookAt(0.0f, 0.3f, -6.0f);
    Vec3 camUp(0.0f, 1.0f, 0.0f);
    float fov_deg = 45.0f;
    float fov = fov_deg * 3.14159265f / 180.0f;
    Vec3 lightPos(5.0f, 8.0f, 0.0f);
    Vec3 lightColor(1.0f,1.0f,0.95f);

    std::vector<Vec3> fb(numPixels);

    float aspect = (float)width / (float)height;
    float halfHeight = std::tan(fov * 0.5f);
    float halfWidth = aspect * halfHeight;
    Vec3 w = (camPos - camLookAt).normalized();
    Vec3 u = cross(camUp, w).normalized();
    Vec3 v = cross(w, u);

    unsigned long long seed = 123456789ULL;
    auto prog_start = std::chrono::high_resolution_clock::now();
    std::cout << "Iniciando renderização de " << width << "x" << height << "...\n" << std::endl;

    auto render_start = std::chrono::high_resolution_clock::now();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            Vec3 pixelColor(0,0,0);
            std::mt19937_64 pixel_rng(seed + idx);
            std::uniform_real_distribution<float> pixel_dist(0.0f, 1.0f);

            for (int s = 0; s < samples; ++s) {
                float rx = pixel_dist(pixel_rng) - 0.5f;
                float ry = pixel_dist(pixel_rng) - 0.5f;
                float px = ((x + 0.5f + rx) / (float)width) * 2.0f - 1.0f;
                float py = ((y + 0.5f + ry) / (float)height) * 2.0f - 1.0f;
                Vec3 dir = (u * (px * halfWidth) + v * (py * halfHeight) - w).normalized();
                Ray ray(camPos, dir);
                pixelColor = pixelColor + trace_ray(ray, spheres, camPos, lightPos, lightColor, max_depth);
            }
            pixelColor = pixelColor / (float)samples;
            pixelColor.x = std::pow(clampf(pixelColor.x, 0.0f, 1.0f), 1.0f/2.2f);
            pixelColor.y = std::pow(clampf(pixelColor.y, 0.0f, 1.0f), 1.0f/2.2f);
            pixelColor.z = std::pow(clampf(pixelColor.z, 0.0f, 1.0f), 1.0f/2.2f);
            fb[idx] = pixelColor;
        }

        float progress = (100.0f * (y + 1)) / height;
        std::cout << "\rProgresso: " << std::fixed << std::setprecision(1)
                  << progress << "% (" << (y+1) << "/" << height << " linhas)" << std::flush;
    }

    std::cout << "\nRender finalizado. Salvando imagem..." << std::endl;

    std::ofstream ofs("../output/output_cpu.ppm");
    ofs << "P3\n" << width << " " << height << "\n255\n";
    for (int j = height - 1; j >= 0; --j) {
        for (int i = 0; i < width; ++i) {
            int idx = j * width + i;
            int ir = (int)(255.99f * clampf(fb[idx].x, 0.0f, 1.0f));
            int ig = (int)(255.99f * clampf(fb[idx].y, 0.0f, 1.0f));
            int ib = (int)(255.99f * clampf(fb[idx].z, 0.0f, 1.0f));
            ofs << ir << " " << ig << " " << ib << "\n";
        }
    }
    ofs.close();

    auto render_end = std::chrono::high_resolution_clock::now();
    std::cout << "Imagem salva como output_cpu.ppm" << std::endl;

    auto prog_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> render_seconds = render_end - render_start;
    std::chrono::duration<double> total_seconds = prog_end - prog_start;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Tempo de render (loop de pixels): " << render_seconds.count() << " s\n";
    std::cout << "Tempo total do programa: " << total_seconds.count() << " s" << std::endl;
    return 0;
}
