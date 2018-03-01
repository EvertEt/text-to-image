import torchfile

import tensorlayer as tl

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "

class_folders = tl.files.load_folder_list('cub_icml/')

for class_folder in class_folders:
    print(class_folder)
    filenames = tl.files.load_file_list(class_folder, regx='\.t7$', printable=False)
    for filename in filenames:
        print(filename)
        file = class_folder + '/' + filename
        with open(file.replace('.t7', '.txt'), 'w') as f:
            data = torchfile.load(file)
            line_count = len(data.char[0])
            for i in range(line_count):
                string = ''
                for j in data.char:
                    string += alphabet[j[i] - 1]
                f.write(string)
                if i != line_count - 1:
                    f.write('\n')