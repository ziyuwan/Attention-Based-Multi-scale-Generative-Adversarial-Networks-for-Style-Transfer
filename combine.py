from PIL import Image
import os


def get_pictures(path):
    images = os.listdir(path)
    names = [os.path.splitext(image)[0] for image in images]
    images = [os.path.join(path, image) for image in images]
    return names, images


content_path = './datasets/contents'
content_names, content_images = get_pictures(content_path)

style_path = './datasets/styles'
style_names, style_images = get_pictures(style_path)

result_path = './results/{}/test_latest/images'
image_big_size = 100
image_small_size = image_big_size - 10


def image_compose():
    to_image = Image.new('RGB', ((len(style_images) + 1) * image_big_size, (len(content_images) + 1) * image_big_size))
    # first row is style images,first column is content images

    # first row
    for x in range(len(style_images)):
        from_image = Image.open(style_images[x]).resize((image_small_size, image_small_size), Image.ANTIALIAS)
        to_image.paste(from_image, ((x + 1) * image_big_size, 0))

    for y in range(len(content_images)):
        from_image = Image.open(content_images[y]).resize((image_small_size, image_small_size), Image.ANTIALIAS)
        to_image.paste(from_image, (0, (y + 1) * image_big_size))
        for x in range(len(style_images)):
            from_image = Image.open(result_image.format(style_names[x], content_names[y])).resize(
                (image_small_size, image_small_size), Image.ANTIALIAS)
            to_image.paste(from_image, ((x + 1) * image_big_size, (y + 1) * image_big_size))
    to_image.save(image_save_path)


models_name = ['AdaIN', 'AdaConv', 'AdaAttN']
models_name = [_ + '_debug' for _ in models_name]
models_name += [_ + '_gan' for _ in models_name]
for model_name in models_name:
    result_image = os.path.join(result_path.format(model_name), '{}_{}_cs.png')
    image_save_path = 'picture/{}.png'.format(model_name)
    image_compose()
