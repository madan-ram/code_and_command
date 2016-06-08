from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from num2words import num2words
import random
from glob import glob
import os, sys
from progress.bar import Bar

def test_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

# get font path using regx
font_dir = os.path.join(sys.argv[1], '*/*.ttf')

print font_dir
# get all the font files
font_path_list = []
for fn in glob(font_dir):
    font_path_list.append(fn)

number_range = (100, 100000)
font_size = 50
sample_size = 5
print "generating fonts"
bar = Bar('Processing', max=number_range[1]-number_range[0])
bar.start()
batch_size = 1000
counter = 0
cur_batch_id = 0
for n in xrange(number_range[0], number_range[1]):
    # randomly sample font to be extracted
    sample_file_path = random.sample(font_path_list, sample_size)
    for i in xrange(len(sample_file_path)):
        if counter%batch_size == 0:
            # test and create batch directory
            cur_batch_id += 1
            test_create_dir(os.path.join('output/text', str(cur_batch_id)))
            test_create_dir(os.path.join('output/number', str(cur_batch_id)))

        font = ImageFont.truetype(sample_file_path[i], font_size)

        text_data = num2words(n, lang='en_IN')
        width, height = font.getsize(text_data)

        img = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0),text_data,(0,0,0),font=font)
        img.save(os.path.join('output/text', str(cur_batch_id), str(n)+'_'+str(i)+'_IMG.jpg'))

        num_data = str(n)
        width, height = font.getsize(num_data)

        img = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0),num_data,(0,0,0),font=font)
        img.save(os.path.join('output/number', str(cur_batch_id), str(n)+'_'+str(i)+'_IMG.jpg'))
        counter += 1
    bar.next()
bar.finish()