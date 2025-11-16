import glob
from PIL import Image
import os

def create_gif(images_path, output_gif_path, frame_duration=100):
    caminhos = sorted(glob.glob(os.path.join(images_path, "*.ppm")))

    if not caminhos:
        print("Nenhuma imagem .ppm encontrada.")
        return

    first = Image.open(caminhos[0]).convert("P", palette=Image.ADAPTIVE)

    frames = []
    for caminho in caminhos[1:]:
        try:
            frame = Image.open(caminho).convert("P", palette=Image.ADAPTIVE)
            frames.append(frame)
        except Exception as e:
            print("Erro:", caminho, e)

    first.save(
        output_gif_path,
        save_all=True,
        append_images=frames,
        duration=frame_duration,
        loop=0,
        optimize=True,
        disposal=2
    )

    print(f"GIF criado com {len(caminhos)} frames → {output_gif_path}")

create_gif(
    images_path="./frames", 
    output_gif_path="animation.gif",
    frame_duration=10
)
