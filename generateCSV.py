import os, csv

def generate_CSV():
    images = []
    labels = []

    for image in os.listdir('./train/'):
        path = '/home/bhappy/Face_Recog/train/' + image
        images.append(path)
    
    for label in images:
        labels.append(label.split('/')[-1].split('.')[0])

    with open('label.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for i in range(len(images)):
            writer.writerow([images[i], labels[i]])
        