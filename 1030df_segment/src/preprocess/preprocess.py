import os

import pandas as pd
from PIL import Image
from tqdm import tqdm


def sample_image_df(base_data_dir, interval=6):
    train_video_list_dir = os.path.join(base_data_dir, 'train_video_list')
    train_video_list = os.listdir(train_video_list_dir)
    train_video_list_df = pd.DataFrame(train_video_list, columns=['train_video_list'])

    image_df_list = []
    for i, row in train_video_list_df.iterrows():
        video_path = os.path.join(train_video_list_dir, row['train_video_list'])
        video_content = pd.read_csv(video_path,
                                    delimiter='\t',
                                    header=None,
                                    names=['train_image', 'train_label_image'])

        video_content['train_image'] = video_content['train_image'].apply(
            lambda x: os.path.basename(x))
        video_content['train_label_image'] = video_content['train_label_image'].apply(
            lambda x: os.path.basename(x))

        video_content.sort_values(by='train_image', ascending=True, inplace=True)

        # 每interval行取1行
        condition = video_content.index % interval == 1
        video_content = video_content[condition]

        image_df_list.append(video_content)

    sample_image_df = pd.concat(image_df_list)

    exist_image_list = os.listdir(os.path.join(base_data_dir, 'train_color'))
    exist_image_df = pd.DataFrame(exist_image_list, columns=['train_image'])

    sample_image_df = pd.merge(sample_image_df, exist_image_df, how='inner')
    return sample_image_df


def process_crop(sample_image_df, process_target=None, image_intput_dir=None, image_output_dir=None, quality=None):
    assert os.path.exists(image_intput_dir)
    assert os.path.exists(image_output_dir)

    image_list = sample_image_df[process_target].tolist()

    image_size = (1024, 1024)
    width, height = image_size
    print('(width, height) = (%s, %s)' % (width, height))

    x_overlaps = [237, 238, 237]
    x_start_list = [width * i - sum(x_overlaps[0:i]) for i in range(4)]
    y_start = 1360
    x_list = [(x_start, x_start + width) for x_start in x_start_list]
    y = (y_start, y_start + height)

    print(x_list)
    print(y)

    for i, image_name in enumerate(tqdm(image_list)):
        with Image.open(os.path.join(image_intput_dir, image_name)) as img:
            for k, x in enumerate(x_list):
                region = img.crop((x[0], y[0], x[1], y[1]))

                (name, ext) = image_name.split('.')
                if process_target == 'train_label_image':
                    l = len('_instanceIds')
                    name = name[0:-l]
                    output_image_name = ''.join([name, '_', str(k), '_instanceIds', '.', ext])
                else:
                    output_image_name = ''.join([name, '_', str(k), '.', ext])
                if quality:
                    region.save(os.path.join(image_output_dir, output_image_name), quality=quality)
                else:
                    region.save(os.path.join(image_output_dir, output_image_name))


if __name__ == '__main__':
    base_data_dir = os.path.expanduser('~/Applications/dev/PycharmProjects/df_vedio_segment/data')
    base_data_dir = os.path.expanduser('~/Documents/ml_data/datafountain')

    sample_image_df = sample_image_df(base_data_dir, interval=6)
    print(sample_image_df.shape)

    print(sample_image_df['train_image'].tolist())

    process_target = 'train_image'
    process_target = 'train_label_image'
    assert process_target in ['train_image', 'train_label_image']

    if process_target == 'train_image':
        image_intput_dir = os.path.join(base_data_dir, 'train_color')
        image_output_dir = os.path.expanduser('~/Applications/dev/PycharmProjects/df_vedio_segment', 'image_output')
    else:
        image_intput_dir = os.path.join(base_data_dir, 'train_label')
        image_output_dir = os.path.expanduser('~/Applications/dev/PycharmProjects/df_vedio_segment', 'label_output')

    process_crop(sample_image_df, process_target=process_target, image_intput_dir=image_intput_dir,
                 image_output_dir=image_output_dir)
