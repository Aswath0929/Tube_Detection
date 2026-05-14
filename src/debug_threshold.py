import cv2
import matplotlib.pyplot as plt


def main():
    image_path = 'data/images/fc1715f8-color.png'
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f'Could not read image: {image_path}')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Histogram to inspect brightness separation
    plt.figure()
    plt.hist(gray.ravel(), bins=256, range=(0, 256))
    plt.title('Grayscale Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    plt.show()

    otsu_thresh, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f'Otsu threshold: {otsu_thresh:.2f}')
    cv2.imshow('Otsu mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
