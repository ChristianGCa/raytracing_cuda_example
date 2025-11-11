import random

def create_random_ppm(width, height, filename):
    with open(filename, 'wb') as f:
        f.write(f"P3\n{width} {height}\n255\n".encode())
        for _ in range(height):
            for _ in range(width):
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                f.write(f"{r} {g} {b}\n".encode())

    print(f"Imagem PPM aleatória salva como {filename}")

if __name__ == "__main__":
    create_random_ppm(256, 256, "../frames/random_image.ppm")

