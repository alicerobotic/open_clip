print('a')
import torch
from PIL import Image
import open_clip
import csv
import numpy as np
from itertools import islice

w0 = ['clear weather']
w1 =['foggy weather']


def test_w(i_t,c_p,w):
    model, _, preprocess = open_clip.create_model_and_transforms('RN50',
    pretrained= c_p )
    # 2024_05_08-22_09_53-model_RN50-lr_0.0005-b_64-j_2-p_amp/epoch_59.pt')
    # 2024_05_08-22_35_09-model_RN50-lr_0.0005-b_64-j_2-p_amp/epoch_58.pt
    tokenizer = open_clip.get_tokenizer('RN50')
    # device = torch.cuda
    # model.to (device=)
    #
    # gtFile = open('/Users/user/PycharmProjects/datasets/GTSRB/test/GT-final_test.csv')
    # gtReader = csv.reader(gtFile, delimiter=';')
    # prefix = '/Users/user/PycharmProjects/datasets/GTSRB/test/images_original/'
    # for row in gtReader:
    #     images.append(plt.imread(prefix + row[0]))  # the 1th column is the filename
    #     labels.append(row[7])  # the 8th column is the label
    # gtFile.close()
    text = tokenizer(['two bumps', 'one bump',  'slippery car',  'curve to The left',  'curve to The right',
          'double curves first to the left',  'double curves first to the right',
          'an adult and a child in a triangular red border',  'cyclists in a triangular red border',  'a cow',  'roadworks',
          'traffic light',  'gated railroad crossing',  'exclamation mark',  'two lines narrowing towards the bottom',
          'road narrows on the left',  'road narrows on the right',  'an arrow merging',  'X',  'a red, inverted triangle',
          'two opposing arrows in red circle',  'stop',  'red circle with a white horizontal line',  'bicycle in red circle',
          'letter t in red circle',  'car in red circle',  'letter m in red circle with two black points horizontal',
          'letter m in red circle with two black points vertical',  'blank white in center',  'red prohibition sign enclosing a left curved arrow',
          'red prohibition sign enclosing a right curved arrow',  'one red and one black cars',  'two digits in circle',  'person and bicycle, vertical',
          'upwards arrow in circle',  'leftwards arrow',  'up and leftwards arrow',  'circular icon with three arrows ',  'bicycle',
          'person and bicycle, horizontal',  'red circular sign with a red diagonal line',  'X sign in red circle',
          'numbers 1 and 15 with red diagonal line',  'numbers 16 and 31 with red diagonal line',  'two opposing arrows in blue background',
          'letter P', 'letter P for disabled',  'letter P for car',  'letter P for van',  'letter P for coach',  'letter P and sidewalk',
          'residential area',  'residential area with red diagonal line',  'upwards arrow in blue rectangle',  ' ’T’ with a red line across the top',
          'roadwork with a red diagonal line,',  'pedestrian crossing',  'cyclist in blue background',  'P and arrow',  'hump in blue background',
          'yellow square with black diagonal line',  'yellow square'])
    text_w = tokenizer(w)

    # text = tokenizer(['detail: two bumps, shape: triangle', 'detail: one bump, shape: triangle', 'detail: slippery car, shape: triangle',
    #                   'detail: curve to The left, shape: triangle', 'detail: curve to The right, shape: triangle',
    #                   'detail: double curves first to the left, shape: triangle', 'detail: double curves first to the right, shape: triangle',
    #                   'detail: an adult and a child in a triangular red border, shape: triangle', 'detail: cyclists in a triangular red border, shape: triangle',
    #                   'detail: a cow, shape: triangle', 'detail: roadworks, shape: triangle', 'detail: traffic light, shape: triangle',
    #                   'detail: gated railroad crossing, shape: triangle', 'detail: exclamation mark, shape: triangle',
    #                   'detail: two lines narrowing towards the bottom, shape: triangle', 'detail: road narrows on the left, shape: triangle',
    #                   'detail: road narrows on the right, shape: triangle', 'detail: an arrow merging, shape: triangle', 'detail: X, shape: triangle',
    #                   'detail: a red, inverted triangle, shape: triangle', 'detail: two opposing arrows in red circle, shape: circle',
    #                   'detail: stop, shape: octagon', 'detail: red circle with a white horizontal line, shape: circle',
    #                   'detail: bicycle in red circle, shape: circle', 'detail: letter t in red circle, shape: circle', 'detail: car in red circle, shape: circle',
    #                   'detail: letter m in red circle with two black points horizontal, shape: circle', 'detail: letter m in red circle with two black points vertical, shape: circle',
    #                   'detail: blank white in center, shape: circle', 'detail: red prohibition sign enclosing a left curved arrow, shape: circle',
    #                   'detail: red prohibition sign enclosing a right curved arrow, shape: circle', 'detail: one red and one black cars, shape: circle',
    #                   'detail: two digits in circle, shape: circle', 'detail: person and bicycle, vertical, shape: circle', 'detail: upwards arrow in circle, shape: circle',
    #                   'detail: leftwards arrow, shape: circle', 'detail: up and leftwards arrow, shape: circle', 'detail: circular icon with three arrows , shape: circle',
    #                   'detail: bicycle, shape: circle', 'detail: person and bicycle, horizontal, shape: circle', 'detail: red circular sign with a red diagonal line, shape: circle',
    #                   'detail: X sign in red circle, shape: circle', 'detail: numbers 1 and 15 with red diagonal line, shape: circle', 'detail: numbers 16 and 31 with red diagonal line, shape: circle',
    #                   'detail: two opposing arrows in blue background, shape: square', 'detail: letter P, shape: square', 'detail: letter P for disabled, shape: square',
    #                   'detail: letter P for car, shape: square', 'detail: letter P for van, shape: square', 'detail: letter P for coach, shape: square',
    #                   'detail: letter P and sidewalk, shape: square', 'detail: residential area, shape: square', 'detail: residential area with red diagonal line, shape: square',
    #                   'detail: upwards arrow in blue rectangle, shape: square', 'detail:  ‘T’ with a red line across the top, shape: square',
    #                   'detail: roadwork with a red diagonal line,, shape: square', 'detail: pedestrian crossing, shape: triangle in square',
    #                   'detail: cyclist in blue background, shape: triangle in square', 'detail: P and arrow, shape: triangle in square',
    #                   'detail: hump in blue background, shape: triangle in square', 'detail: yellow square with black diagonal line, shape: square',
    #                   'detail: yellow square, shape: square'])
    image = []
    label = []
    count = 0
    su =0
    i = 0

    for c in range(0, 62):
        # prefix = rootpath + '/' + format(c, '05d') + '/' + 'new/'  # subdirectory for class
        # os.chdir(prefix)
        gtFile = open('/Users/user/Downloads/BelguimTS/' + i_t + '/' + format(c, '05d') + '/' + 'GT-' + format(c, '05d') + '.csv')  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file

        # # loop over all images in current annotations file
        # ii = 0
        for row in islice(gtReader, 1, None):
            pre = '/Users/user/Downloads/BelguimTS/' + i_t + '/' + format(c, '05d') + '/' + row[0]
            im = preprocess(Image.open(pre))
            image.append(im)

            la = row[7]
            label.append(int(la))
            i = i + 1

            # if i % 512 == 1 or i == 2534:
            if i % 512 == 0 or i == 2534:
                print(i)

                with torch.no_grad(), torch.cuda.amp.autocast():
                    image = np.array(image)
                    image = torch.tensor(image)
                    label = np.array(label)
                    label = torch.tensor(label)
                    image_features = model.encode_image(image)
                    text_features = model.encode_text(text)
                    text_w_features = model.encode_text(text_w)

                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    # print(image_features[0].shape)
                    # print(image_features[0])
                    # print(torch.norm(image_features[0],p=2))
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    # print(text_features[0].shape)
                    # print(text_features[0])
                    # print(torch.norm(text_features[0],p=2))
                    text_w_features /= text_w_features.norm(dim=-1, keepdim=True)
                    # print(text_w_features[0].shape)
                    # print(text_w_features[0])
                    # print(torch.norm(text_w_features[0],p=2))
                    image_features = image_features + text_w_features
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    # print(torch.norm(image_features[0], p=2))

                    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    # print(text_probs.shape)
                    pred = torch.argmax(text_probs, dim=1)
                    #
                    # print(pred, label)
                    re = (pred == label)
                    # print(re)
                    # print(sum(re))
                    count += sum(re)

                    if i != 2534:
                        su += 512
                    else:
                        su = su + (i % 512)
                    print(su)
                    print('acc:',count / su)

                image = []
                label = []
            # break
            # if i > 1000:
            #     break
            # if su == 512:
            #     break
        # break
        gtFile.close()


# test('Testing','2024_05_08-22_09_53-model_RN50-lr_0.0005-b_64-j_2-p_amp/checkpoints/epoch_59.pt')
# test('Testing_foggy','2024_05_08-22_09_53-model_RN50-lr_0.0005-b_64-j_2-p_amp/checkpoints/epoch_59.pt')

# test('Testing','2024_05_17-13_30_46-model_RN50-lr_0.0005-b_64-j_2-p_amp/checkpoints/epoch_124.pt')
# test('Testing_foggy','2024_05_17-13_30_46-model_RN50-lr_0.0005-b_64-j_2-p_amp/checkpoints/epoch_124.pt')

#
# test_w('Testing','2024_05_17-13_30_46-model_RN50-lr_0.0005-b_64-j_2-p_amp/checkpoints/epoch_124.pt',w0)
# test_w('Testing_foggy','2024_05_17-13_30_46-model_RN50-lr_0.0005-b_64-j_2-p_amp/checkpoints/epoch_124.pt',w1)
#

def test(i_t,c_p):
    model, _, preprocess = open_clip.create_model_and_transforms('RN50',
    pretrained= c_p )
    # 2024_05_08-22_09_53-model_RN50-lr_0.0005-b_64-j_2-p_amp/epoch_59.pt')
    # 2024_05_08-22_35_09-model_RN50-lr_0.0005-b_64-j_2-p_amp/epoch_58.pt
    tokenizer = open_clip.get_tokenizer('RN50')
    # device = torch.cuda
    # model.to (device=)
    #
    # gtFile = open('/Users/user/PycharmProjects/datasets/GTSRB/test/GT-final_test.csv')
    # gtReader = csv.reader(gtFile, delimiter=';')
    # prefix = '/Users/user/PycharmProjects/datasets/GTSRB/test/images_original/'
    # for row in gtReader:
    #     images.append(plt.imread(prefix + row[0]))  # the 1th column is the filename
    #     labels.append(row[7])  # the 8th column is the label
    # gtFile.close()
    text = tokenizer(['two bumps', 'one bump',  'slippery car',  'curve to The left',  'curve to The right',
          'double curves first to the left',  'double curves first to the right',
          'an adult and a child in a triangular red border',  'cyclists in a triangular red border',  'a cow',  'roadworks',
          'traffic light',  'gated railroad crossing',  'exclamation mark',  'two lines narrowing towards the bottom',
          'road narrows on the left',  'road narrows on the right',  'an arrow merging',  'X',  'a red, inverted triangle',
          'two opposing arrows in red circle',  'stop',  'red circle with a white horizontal line',  'bicycle in red circle',
          'letter t in red circle',  'car in red circle',  'letter m in red circle with two black points horizontal',
          'letter m in red circle with two black points vertical',  'blank white in center',  'red prohibition sign enclosing a left curved arrow',
          'red prohibition sign enclosing a right curved arrow',  'one red and one black cars',  'two digits in circle',  'person and bicycle, vertical',
          'upwards arrow in circle',  'leftwards arrow',  'up and leftwards arrow',  'circular icon with three arrows ',  'bicycle',
          'person and bicycle, horizontal',  'red circular sign with a red diagonal line',  'X sign in red circle',
          'numbers 1 and 15 with red diagonal line',  'numbers 16 and 31 with red diagonal line',  'two opposing arrows in blue background',
          'letter P', 'letter P for disabled',  'letter P for car',  'letter P for van',  'letter P for coach',  'letter P and sidewalk',
          'residential area',  'residential area with red diagonal line',  'upwards arrow in blue rectangle',  ' ’T’ with a red line across the top',
          'roadwork with a red diagonal line,',  'pedestrian crossing',  'cyclist in blue background',  'P and arrow',  'hump in blue background',
          'yellow square with black diagonal line',  'yellow square'])
    # text_w = tokenizer(w)


    image = []
    label = []
    count = 0
    su =0
    i = 0

    for c in range(0, 62):
        # prefix = rootpath + '/' + format(c, '05d') + '/' + 'new/'  # subdirectory for class
        # os.chdir(prefix)
        gtFile = open('/Users/user/Downloads/BelguimTS/' + i_t + '/' + format(c, '05d') + '/' + 'GT-' + format(c, '05d') + '.csv')  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file

        # # loop over all images in current annotations file
        # ii = 0
        for row in islice(gtReader, 1, None):
            pre = '/Users/user/Downloads/BelguimTS/' + i_t + '/' + format(c, '05d') + '/' + row[0]
            im = preprocess(Image.open(pre))
            image.append(im)

            la = row[7]
            label.append(int(la))
            i = i + 1

            if i % 512 == 0 or i == 2534:
                print(i)

                with torch.no_grad(), torch.cuda.amp.autocast():
                    image = np.array(image)
                    image = torch.tensor(image)
                    label = np.array(label)
                    label = torch.tensor(label)
                    image_features = model.encode_image(image)
                    text_features = model.encode_text(text)
                    # text_w_features = model.encode_text(text_w)

                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    print(image_features.shape)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    print(text_features)
                    # text_w_features /= text_w_features.norm(dim=-1, keepdim=True)
                    # print(text_w_features.shape)
                    # image_features = image_features + text_w_features

                    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    print(text_probs.shape)
                    pred = torch.argmax(text_probs, dim=1)
                    #
                    # print(pred, label)
                    re = (pred == label)
                    # print(re)
                    # print(sum(re))
                    count += sum(re)

                    if i != 2534:
                        su += 512
                    else:
                        su = su + (i % 512)
                    print(su)
                    print('acc:',count / su)

                image = []
                label = []
            # if i > 1000:
            #     break
            # if su == 512:
            #     break
        gtFile.close()

# test('Testing','2024_05_17-13_30_46-model_RN50-lr_0.0005-b_64-j_2-p_amp/checkpoints/epoch_124.pt')
# test('Testing_foggy','2024_05_17-13_30_46-model_RN50-lr_0.0005-b_64-j_2-p_amp/checkpoints/epoch_124.pt')

# test('Testing','/Users/user/PycharmProjects/NIPS/open_clip/src/training/logs/2024_05_17-22_46_14-model_RN50-lr_0.0005-b_64-j_2-p_amp/checkpoints/epoch_122.pt')
# test('Testing_foggy','/Users/user/PycharmProjects/NIPS/open_clip/src/training/logs/2024_05_17-22_46_14-model_RN50-lr_0.0005-b_64-j_2-p_amp/checkpoints/epoch_122.pt')
#
# test_w('Testing','/Users/user/PycharmProjects/NIPS/open_clip/src/training/logs/2024_05_17-22_46_14-model_RN50-lr_0.0005-b_64-j_2-p_amp/checkpoints/epoch_122.pt',w0)
# test_w('Testing_foggy','/Users/user/PycharmProjects/NIPS/open_clip/src/training/logs/2024_05_17-22_46_14-model_RN50-lr_0.0005-b_64-j_2-p_amp/checkpoints/epoch_122.pt',w1)


# test('Testing','/Users/user/PycharmProjects/icon/open_clip/src/training/logs/2024_05_17-21_11_10-model_RN50-lr_0.0005-b_64-j_2-p_amp/checkpoints/epoch_110.pt')
# test('Testing_foggy','/Users/user/PycharmProjects/icon/open_clip/src/training/logs/2024_05_17-21_11_10-model_RN50-lr_0.0005-b_64-j_2-p_amp/checkpoints/epoch_110.pt')


test_w('Testing','/Users/user/PycharmProjects/icon/open_clip/src/training/logs/2024_05_17-21_11_10-model_RN50-lr_0.0005-b_64-j_2-p_amp/checkpoints/epoch_110.pt',w0)
test_w('Testing_foggy','/Users/user/PycharmProjects/icon/open_clip/src/training/logs/2024_05_17-21_11_10-model_RN50-lr_0.0005-b_64-j_2-p_amp/checkpoints/epoch_110.pt',w1)

# test('Testing','/Users/user/PycharmProjects/NIPS/open_clip/src/training/logs/2024_05_18-01_06_34-model_RN50-lr_0.0005-b_64-j_2-p_amp/checkpoints/epoch_87.pt')
# test('Testing_foggy','/Users/user/PycharmProjects/NIPS/open_clip/src/training/logs/2024_05_18-01_06_34-model_RN50-lr_0.0005-b_64-j_2-p_amp/checkpoints/epoch_87.pt')
