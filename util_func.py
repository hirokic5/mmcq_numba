from PIL import Image

def color2tile(color_info, ratio="equal"):
    
    IMG_SIZE = 64
    MARGIN = 0
    width = IMG_SIZE * len(color_info) + MARGIN * 2
    height = IMG_SIZE + MARGIN * 2
    tiled_color_img = Image.new(
        mode='RGB', size=(width, height), color='#333333')
    color_sum = sum([c[0] for c in color_info])
    count_accum = 0
    for i,info in enumerate(color_info):
        count, color = info
        color_hex_str = '#%02x%02x%02x' % tuple(color)
        IMG_SIZE_width = IMG_SIZE if ratio=="equal" else int(width * count / color_sum)
        start = IMG_SIZE * i if ratio=="equal" else int(width * count_accum / color_sum)
        
        
        color_img = Image.new(
            mode='RGB', size=(IMG_SIZE_width, IMG_SIZE),
            color=color_hex_str)
        tiled_color_img.paste(
            im=color_img,
            box=(MARGIN + start, MARGIN))
        count_accum += count
        
    return tiled_color_img