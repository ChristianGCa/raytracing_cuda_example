// Run:
// ./raytrace_improved
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include <iomanip>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>

struct Vec3 {
    float x,y,z;
    __host__ __device__ Vec3(): x(0),y(0),z(0) {}
    __host__ __device__ Vec3(float a,float b,float c): x(a),y(b),z(c) {}
    __host__ __device__ Vec3 operator+(const Vec3 &o) const { return Vec3(x+o.x,y+o.y,z+o.z); }
    __host__ __device__ Vec3 operator-(const Vec3 &o) const { return Vec3(x-o.x,y-o.y,z-o.z); }
    __host__ __device__ Vec3 operator*(float s) const { return Vec3(x*s,y*s,z*s); }
    __host__ __device__ Vec3 operator*(const Vec3 &o) const { return Vec3(x*o.x,y*o.y,z*o.z); }
    __host__ __device__ Vec3 operator/(float s) const { float inv = 1.0f/s; return Vec3(x*inv,y*inv,z*inv); }
    __host__ __device__ float length() const { return sqrtf(x*x + y*y + z*z); }
    __host__ __device__ Vec3 normalized() const { float l = length(); return l>0 ? (*this)/l : Vec3(0,0,0); }
};

__host__ __device__ inline float dot(const Vec3 &a,const Vec3 &b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ inline Vec3 cross(const Vec3 &a,const Vec3 &b){
    return Vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
__host__ __device__ inline Vec3 reflect(const Vec3 &I,const Vec3 &N){ return I - N*(2.0f*dot(I,N)); }
__host__ __device__ inline float clampf(float v,float a,float b){ return v<a?a:(v>b?b:v); }

struct Ray { Vec3 o,d; __host__ __device__ Ray(){} __host__ __device__ Ray(const Vec3 &o_, const Vec3 &d_): o(o_), d(d_.normalized()) {} };

struct Material {
    Vec3 albedo;
    float reflectivity; // 0..1
    float specular_exponent; // phong exponent
    float absorption; // attenuation coefficient
    __host__ __device__ Material(): albedo(1,1,1), reflectivity(0), specular_exponent(8), absorption(0) {}
    __host__ __device__ Material(const Vec3 &a,float r,float s,float abs): albedo(a), reflectivity(r), specular_exponent(s), absorption(abs) {}
};

struct Sphere {
    Vec3 center; float radius; Material mat;
    __host__ __device__ Sphere() {}
    __host__ __device__ Sphere(const Vec3 &c,float r,const Material &m): center(c), radius(r), mat(m) {}
    __host__ __device__ bool intersect(const Ray &ray, float &tHit, Vec3 &n) const {
        Vec3 oc = ray.o - center;
        float a = dot(ray.d, ray.d);
        float b = 2.0f * dot(oc, ray.d);
        float c = dot(oc, oc) - radius*radius;
        float disc = b*b - 4*a*c;
        if (disc < 0) return false;
        float sq = sqrtf(disc);
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

__device__ Vec3 background_color(const Vec3 &dir) {
    float t = 0.5f * (dir.y + 1.0f);
    Vec3 c1(0.7f,0.9f,1.0f), c2(0.15f,0.16f,0.2f);
    return c1 * t + c2 * (1.0f - t);
}

__global__ void init_rand_kernel(curandState *randStates, int width, int height, unsigned long long seed){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=width || y>=height) return;
    int idx = y*width + x;
    curand_init(seed + idx, 0, 0, &randStates[idx]);
}

__global__ void render_kernel(Vec3 *fb, int width, int height,
                              Sphere *spheres, int nspheres,
                              Vec3 camPos, Vec3 camLookAt, Vec3 camUp,
                              float fov_deg, int max_depth, int samples,
                              Vec3 lightPos, Vec3 lightColor, curandState *randStates)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=width || y>=height) return;
    int idx = y*width + x;

    curandState localState = randStates[idx];

    float aspect = float(width)/float(height);
    float fov = fov_deg * 3.14159265f / 180.0f;
    float halfHeight = tanf(fov*0.5f);
    float halfWidth = aspect * halfHeight;

    Vec3 w = (camPos - camLookAt).normalized();
    Vec3 u = cross(camUp, w).normalized();
    Vec3 v = cross(w, u);

    Vec3 pixelColor(0,0,0);

    for (int s = 0; s < samples; ++s) {
        float rx = curand_uniform(&localState) - 0.5f;
        float ry = curand_uniform(&localState) - 0.5f;
        float px = ( (x + 0.5f + rx) / float(width) ) * 2.0f - 1.0f;
        float py = ( (y + 0.5f + ry) / float(height) ) * 2.0f - 1.0f;
        Vec3 dir = (u * (px * halfWidth) + v * (py * halfHeight) - w).normalized();

        Ray ray(camPos, dir);
        Vec3 throughput(1,1,1);
        Vec3 radiance(0,0,0);

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

            if (mat.absorption > 0.0f) {
                float atten = expf(-mat.absorption * bestT);
                throughput = throughput * atten;
            }

            Vec3 toLight = (lightPos - hitP);
            float distToLight = toLight.length();
            Vec3 L = toLight / distToLight;
            Ray shadowRay(hitP + N * 1e-4f, L);
            bool inShadow = false;
            for (int i=0;i<nspheres;++i){
                float tShadow; Vec3 tmpN;
                if (spheres[i].intersect(shadowRay, tShadow, tmpN)) {
                    if (tShadow < distToLight - 1e-4f) { inShadow = true; break; }
                }
            }

            float NdotL = fmaxf(dot(N, L), 0.0f);
            Vec3 diffuse = mat.albedo * NdotL;

            Vec3 V = (camPos - hitP).normalized();
            Vec3 H = (V + L).normalized();
            float NdotH = fmaxf(dot(N, H), 0.0f);
            float spec = powf(NdotH, mat.specular_exponent);
            Vec3 specular = lightColor * spec * mat.reflectivity;

            Vec3 direct(0,0,0);
            if (!inShadow) direct = (diffuse * lightColor) + specular;

            Vec3 ambient = mat.albedo * 0.05f;
            float nonRef = 1.0f - mat.reflectivity;
            radiance = radiance + throughput * ( (direct + ambient) * nonRef );

            if (mat.reflectivity > 0.0f) {
                Vec3 R = reflect(ray.d, N).normalized();
                ray.o = hitP + R * 1e-4f;
                ray.d = R;
                throughput = throughput * mat.albedo * mat.reflectivity;
                if (throughput.x < 1e-4f && throughput.y < 1e-4f && throughput.z < 1e-4f) break;
            } else {
                break;
            }
        }
        pixelColor = pixelColor + radiance;
    }

    pixelColor = pixelColor / float(samples);
    pixelColor.x = powf(clampf(pixelColor.x, 0.0f, 1.0f), 1.0f/2.2f);
    pixelColor.y = powf(clampf(pixelColor.y, 0.0f, 1.0f), 1.0f/2.2f);
    pixelColor.z = powf(clampf(pixelColor.z, 0.0f, 1.0f), 1.0f/2.2f);

    fb[idx] = pixelColor;
    randStates[idx] = localState;
}

void save_ppm(const char *filename, const Vec3 *h_fb, int width, int height) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Erro ao criar o arquivo: %s\n", filename);
        return;
    }

    fprintf(f, "P3\n%d %d\n255\n", width, height);
    for (int j = height - 1; j >= 0; --j) {
        for (int i = 0; i < width; ++i) {
            int idx = j * width + i;
            int ir = int(255.99f * clampf(h_fb[idx].x, 0.0f, 1.0f));
            int ig = int(255.99f * clampf(h_fb[idx].y, 0.0f, 1.0f));
            int ib = int(255.99f * clampf(h_fb[idx].z, 0.0f, 1.0f));
            fprintf(f, "%d %d %d\n", ir, ig, ib);
        }
    }
    fclose(f);
}

int main(){
    const int width = 1920;
    const int height = 1080;
    const int numPixels = width * height;
    const int numFrames = 60;

    Vec3 *d_fb;
    cudaMalloc((void**)&d_fb, numPixels * sizeof(Vec3));
    Vec3 *h_fb = new Vec3[numPixels];

    const int nspheres = 6;
    Sphere h_spheres[nspheres];
    h_spheres[0] = Sphere(Vec3(0.0f, -10005.0f, -20.0f), 10000.0f, Material(Vec3(0.8f,0.8f,0.8f), 0.0f, 16.0f, 0.01f));
    h_spheres[1] = Sphere(Vec3(-2.5f, 0.5f, -10.0f), 1.5f, Material(Vec3(0.95f,0.95f,0.98f), 0.95f, 64.0f, 0.02f));
    h_spheres[2] = Sphere(Vec3(1.8f, -0.2f, -7.5f), 1.3f, Material(Vec3(0.9f,0.4f,0.35f), 0.12f, 32.0f, 0.45f));
    h_spheres[3] = Sphere(Vec3(0.0f, 1.2f, -6.0f), 1.2f, Material(Vec3(0.2f,0.9f,0.35f), 0.6f, 32.0f, 0.08f));
    h_spheres[4] = Sphere(Vec3(3.5f, 0.3f, -13.0f), 1.0f, Material(Vec3(0.6f,0.7f,0.95f), 0.7f, 64.0f, 0.03f));

    Sphere *d_spheres;
    cudaMalloc((void**)&d_spheres, nspheres * sizeof(Sphere));

    Vec3 camPos(0.0f, 1.0f, 2.5f);
    Vec3 camLookAt(0.0f, 0.3f, -6.0f);
    Vec3 camUp(0.0f, 1.0f, 0.0f);
    float fov = 45.0f;
    int max_depth = 6;
    int samples = 30;
    Vec3 lightPos(5.0f, 8.0f, 0.0f);
    Vec3 lightColor(1.0f,1.0f,0.95f);

    curandState *d_randStates;
    cudaMalloc((void**)&d_randStates, numPixels * sizeof(curandState));

    dim3 block(16,16);
    dim3 grid( (width + block.x - 1)/block.x, (height + block.y - 1)/block.y );

    unsigned long long seed = 123456789ULL;
    init_rand_kernel<<<grid, block>>>(d_randStates, width, height, seed);
    cudaDeviceSynchronize();

    auto prog_start = std::chrono::high_resolution_clock::now();

    // prepare CUDA events for per-frame GPU timing
    cudaEvent_t fstart, fstop;
    cudaEventCreate(&fstart);
    cudaEventCreate(&fstop);

    std::vector<float> frame_kernel_ms;
    frame_kernel_ms.reserve(numFrames);

    Vec3 initialCenter1 = h_spheres[1].center;
    Vec3 initialCenter2 = h_spheres[2].center;
    Vec3 initialCenter3 = h_spheres[3].center;

    const float moveXPerFrame = 0.05f; 
    const float moveYPerFrame = 0.03f; 
    const float moveZPerFrame = 0.06f; 

    for (int frame = 0; frame < numFrames; ++frame) {
        float f = float(frame);

        // 🔹 Movimento da esfera reflexiva (em Y e Z)
        float currentY1 = initialCenter1.y + f * moveYPerFrame;
        float currentZ1 = initialCenter1.z + f * moveZPerFrame;
        h_spheres[1].center = Vec3(initialCenter1.x, currentY1, currentZ1);

        // 🔹 Movimento vertical senoidal da vermelha
        float currentY2 = initialCenter2.y + sinf(f * 0.1f) * 0.8f;
        h_spheres[2].center = Vec3(initialCenter2.x, currentY2, initialCenter2.z);

        // 🔹 Movimento lateral da verde
        float currentX3 = initialCenter3.x + f * moveXPerFrame;
        h_spheres[3].center = Vec3(currentX3, initialCenter3.y, initialCenter3.z);
        
    cudaMemcpy(d_spheres, h_spheres, nspheres * sizeof(Sphere), cudaMemcpyHostToDevice);

    // measure GPU kernel time per frame
    cudaEventRecord(fstart);
    render_kernel<<<grid, block>>>(d_fb, width, height, d_spheres, nspheres,
                       camPos, camLookAt, camUp, fov, max_depth, samples,
                       lightPos, lightColor, d_randStates);
    cudaDeviceSynchronize();
    cudaEventRecord(fstop);
    cudaEventSynchronize(fstop);
    float frame_ms = 0.0f;
    cudaEventElapsedTime(&frame_ms, fstart, fstop);
    frame_kernel_ms.push_back(frame_ms);

    cudaMemcpy(h_fb, d_fb, numPixels * sizeof(Vec3), cudaMemcpyDeviceToHost);

    char filename[100];
    snprintf(filename, sizeof(filename), "../frames/frame_cuda_%02d.ppm", frame);
    save_ppm(filename, h_fb, width, height);

    std::cout << "\rRenderizando Frame: " << std::fixed << std::setprecision(0)
          << frame + 1 << "/" << numFrames << " -> " << filename << std::flush;
    }

    std::cout << "\rRenderizando Frame: " << numFrames << "/" << numFrames << ". Concluído!\n";

    delete[] h_fb;
    cudaFree(d_fb);
    cudaFree(d_spheres);
    cudaFree(d_randStates);

    // compute timing summary
    auto prog_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_seconds = prog_end - prog_start;

    // compute stats for GPU kernel ms per frame
    double sum_ms = 0.0;
    double min_ms = 1e300, max_ms = 0.0;
    for (float v : frame_kernel_ms) {
        sum_ms += v;
        if (v < min_ms) min_ms = v;
        if (v > max_ms) max_ms = v;
    }
    double avg_ms = (frame_kernel_ms.empty() ? 0.0 : sum_ms / frame_kernel_ms.size());

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nFrames renderizados: " << numFrames << "\n";
    std::cout << "Tempo total (programa): " << total_seconds.count() << " s\n";
    std::cout << "Tempo médio por kernel GPU: " << (avg_ms/1000.0) << " s (" << avg_ms << " ms)\n";
    std::cout << "Min/Max kernel time: " << (min_ms/1000.0) << " s / " << (max_ms/1000.0) << " s\n";

    // destroy events
    cudaEventDestroy(fstart);
    cudaEventDestroy(fstop);

    printf("Todos os frames renderizados.\n");

    //char *args[]={"python3 ../scripts/create_gif.py",NULL};
    //execvp(args[0],args);

    return 0;
}
