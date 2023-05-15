from face_swap.image_preprocessing import align
from face_swap.utils import read_image, save_image
from face_swap.swap import merge


def run(
        source_path: str,
        target_path: str,
        output_path: str
):
    source_img = read_image(source_path)
    target_img = read_image(target_path)

    s_aligned, s_ldmks = align(source_img)
    t_aligned, t_ldmks = align(target_img)

    merged_face = merge(s_aligned, s_ldmks, t_aligned, t_ldmks)
    
    save_image(merged_face, output_path)


if __name__ == '__main__':
    source_img_path = "data/f1.jpg"
    target_img_path = "data/macron2.jpg"
    output_img_path = "output/merged.jpg"

    run(source_img_path, target_img_path, output_img_path)