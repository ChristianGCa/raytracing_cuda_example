import glob
from PIL import Image
import os

def create_gif(images_path, output_gif_path, frame_duration=100):
    caminhos_arquivos = sorted(glob.glob(os.path.join(images_path, '*.ppm')))
    
    if not caminhos_arquivos:
        print(f"Nenhuma imagem .ppm encontrada na pasta: {images_path}")
        return

    imagens = []
    for caminho in caminhos_arquivos:
        try:
            img = Image.open(caminho)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            imagens.append(img)
        except IOError:
            print(f"Erro ao abrir a imagem {caminho}")

    if not imagens:
        print("Nenhuma imagem válida foi carregada. Não foi possível criar o GIF.")
        return

    imagens[0].save(
        output_gif_path,
        save_all=True,
        append_images=imagens[1:],
        duration=frame_duration,
        loop=0
    )
    
    print(f"GIF processed: {output_gif_path} with {len(imagens)} frames.")

create_gif(
    images_path="./frames", 
    output_gif_path="animation.gif",
    frame_duration=10
)
