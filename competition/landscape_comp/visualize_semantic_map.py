from pathlib import Path

import numpy as np
import PIL.Image as Image
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm

classes = ["mountain", "sky", "water", "sea", "rock",
           "tree", "earth", "hill", "river", "sand",
           "land","building", "grass", "plant", "person",
           "boat", "waterfall", "wall", "pier", "path",
           "lake", "bridge", "field", "road", "railing",
           "fence", "ship", "house", "other"]
classes_dict = dict( enumerate(classes) )

# refered to https://github.com/open-mmlab/mmsegmentation/blob/7553fbe948dd258400cd574feb66a1d6e1316d8a/mmseg/models/segmentors/base.py#L217
def show_result(img_path, map_path, n_classes=29, palette=None, opacity=0.5):
    '''
    img, map: Path
    palette: numpy.NDarray
    n_classes, opacity: int
    '''
    map = Image.open(map_path)
    # 尽量不resize semantic map的大小
    img = Image.open(img_path).resize(map.size)
    
    img_array = np.array(img)
    map_array = np.array(map)

    if palette is None:
        state = np.random.get_state()
        np.random.seed(42)
        # random palette
        palette = np.random.randint(
            0, 256, size=(n_classes, 3))
        np.random.set_state(state)
    
    assert palette.shape[0] == n_classes
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    # assert 0 <= opacity <= 1.0
    
    color_seg = np.zeros((map_array.shape[0], map_array.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[map_array == label, :] = color
        
    # convert to BGR
    # color_seg = color_seg[..., ::-1]

    img_array = img_array * (1 - opacity) + color_seg * opacity
    # float 转换为 uint8
    img_array = img_array.astype(np.uint8)
    res = Image.fromarray(img_array).convert("RGB")

    return res


def main():
    '''
    root: Path
    # 原图目录，分割图目录，结果目录
    imgs_dir, maps_dir, res_dir: Path
    '''
    root = Path(r"./train")
    imgs_dir = root / "imgs"
    maps_dir = root / "labels"
    res_dir = root / "res"
    
    n_classes = 29
    palette = np.random.randint(0, 256, size=(n_classes, 3))
    opacity = 2.0
    
    # classes_dict可视化
    colors_array = np.ones((n_classes*30, 768, 3)) * 255.0
    for label, color in enumerate(palette):
        start_row = 30*label
        colors_array[ start_row:start_row+30, 150: ] = color
        # 最左边写类别名字，cv2的坐标原点在左上角，x为w，y为h
        cv.putText(colors_array, classes_dict[label], (0, start_row+22), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 1)

    colors_array = colors_array.astype(np.uint8)
    Image.fromarray(colors_array).save( root / "colors_map.png" )

    if not res_dir.exists():
        res_dir.mkdir()
    for img_path, map_path in tqdm(zip(imgs_dir.iterdir(), maps_dir.iterdir())):
        # opacity 不透明度为1，只可视化语义分割的标签图
        res = show_result(img_path, map_path, n_classes, palette, opacity)
        res.save(res_dir / map_path.name)


if __name__ == "__main__":
    main()
    
