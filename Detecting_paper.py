import cv2
import numpy as np
import argparse
import os
import sys


def load_and_preprocess_image(path):
    """
    Загрузка и предварительная обработка изображения
    Args:
        path (str): Путь к файлу изображения для загрузки

    Returns:
        - original: Исходное изображение в формате BGR
        - edges: Бинарное изображение с границами после обработки детектором Canny

    Raises:
        FileNotFoundError - Если файл не существует.
        IsADirectoryError - Если указанный путь является директорией, а не файлом.
        IOError - Если не удалось загрузить изображение.
    """

    if not os.path.exists(path):
        print(f"Ошибка: Файл '{path}' не существует.")
        sys.exit(1)

    if not os.path.isfile(path):
        print(f"Ошибка: '{path}' является директорией, а не файлом.")
        sys.exit(1)

    # Загрузка изображения
    image = cv2.imread(path)
    if image is None:
        print(f"Ошибка: Не удалось загрузить изображение '{path}'.")
        sys.exit(1)

    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение детектора границ Canny
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    return original, edges


def find_document_contour(edges):
    """
    Поиск контура документа на изображении
    Args:
        edges : Бинарное изображение с границами после детектора Canny

    Returns:
            - best_quadrilateral: Массив точек найденного четырехугольника
            - max_area: Площадь найденного контура в пикселях
    """
    # Поиск контуров
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Поиск четырёхугольника с наибольшей площадью
    max_area = 0
    best_quadrilateral = None

    for contour in contours:
        # Аппроксимация контура многоугольником
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Проверяем, что это четырёхугольник
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area and area > 500:  # Минимальная площадь 500 пикселей для проверки
                max_area = area
                best_quadrilateral = approx
    return best_quadrilateral, max_area


def visualize_results(original, best_quadrilateral, max_area):
    """
    Визуализация и вывод результатов обнаружения документа
    Args:
        original: Исходное изображение в формате BGR
        best_quadrilateral: Массив точек найденного четырехугольника
        max_area: Площадь найденного контура в пикселя

    Returns:
        result: Изображение с визуализацией результатов или None при ошибке
    """
    if best_quadrilateral is None:
        print("Документ на изображении не найден")
        return None

    # Рисование найденного четырёхугольника
    result = original.copy()
    cv2.drawContours(result, [best_quadrilateral], -1, (0, 255, 0), 3)

    # Отрисовка вершин для наглядности
    for i, point in enumerate(best_quadrilateral):
        x, y = point[0]
        cv2.circle(result, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(result, str(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Вывод координат
    coordinates = best_quadrilateral.reshape(4, 2)
    print("Координаты углов документа:")
    for i, (x, y) in enumerate(coordinates):
        print(f"Угол {i + 1}: ({x}, {y})")

    print(f"Площадь документа: {max_area:.2f} пикселей")

    return result


def main():
    """
    Основная функция программы для обнаружения прямоугольного документа на изображении

    Обрабатывает аргументы командной строки, загружает изображение,
    применяет корректировки и отображает результаты.
    """

    parser = argparse.ArgumentParser(description='Обнаружение документа на изображении')
    parser.add_argument('--input', help='Путь к входному изображению(по умолчанию: image.jpg)')
    parser.add_argument('-o', '--output', default='result.jpg',
                        help='Путь для сохранения результата (по умолчанию: result.jpg)')

    args = parser.parse_args()

    # Получение пути к изображению
    if args.input:
        path = args.input
    else:
        path = input("Введите путь к изображению: ")

    # Проверка существования файла
    if not os.path.exists(path):
        print(f"Ошибка: файл '{path}' не существует.")
        return None

    # Загрузка и предобработка изображения
    preprocessing_result = load_and_preprocess_image(path)
    if preprocessing_result is None:
        return None
    original, edges = preprocessing_result

    # Поиск контура документа
    best_quadrilateral, max_area = find_document_contour(edges)

    # Визуализация результатов
    result1 = visualize_results(original, best_quadrilateral, max_area)

    if result1 is not None:
        cv2.imshow("Обнаруженный документ", result1)
        cv2.imwrite(args.output, result1)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
