import argparse
import os
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

# Укажите путь к Tesseract (если он используется)
import pytesseract

# Путь к папке с изображениями
IMAGE_DIR = "/Users/almas/Desktop/car/img"

# Код для распознавания изображений
@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # модель
        source=IMAGE_DIR,  # путь к папке с изображениями
        imgsz=(640, 640),  # размер изображения
        conf_thres=0.25,  # порог уверенности
        iou_thres=0.45,  # порог IoU для NMS
        max_det=1000,  # максимальное количество детекций
        device='',  # устройство CUDA
        view_img=False,  # показывать результаты
        save_txt=False,  # сохранять результаты в *.txt
        save_conf=False,  # сохранять уверенности в метках
        save_crop=False,  # сохранять обрезанные боксы
        nosave=False,  # не сохранять изображения/видео
        classes=None,  # фильтровать по классам
        agnostic_nms=False,  # класс-агностичный NMS
        augment=False,  # увеличенная инференция
        visualize=False,  # визуализировать признаки
        update=False,  # обновить модели
        project=ROOT / 'runs/detect',  # сохранить результаты
        name='exp',  # имя папки для сохранения
        exist_ok=False,  # существующая папка
        line_thickness=3,  # толщина рамки
        hide_labels=False,  # скрыть метки
        hide_conf=False,  # скрыть уверенности
        half=False,  # использовать FP16
        dnn=False,  # использовать OpenCV DNN для ONNX
        ):
    source = str(source)  # Преобразование пути в строку
    save_img = not nosave and not source.endswith('.txt')  # сохранять изображения

    # Директории
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # создаем папку для сохранения
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # создание папки для меток

    # Загрузка модели
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # проверка размера изображения

    # Полезные настройки
    half &= (pt or jit or engine) and device.type != 'cpu'  # FP16 только на CUDA

    # Загрузка изображений
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # размер батча

    # Запуск инференции
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # прогрев

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # преобразование типов
        im /= 255  # нормализация
        if len(im.shape) == 3:
            im = im[None]  # расширяем размерность

        # Инференция
        pred = model(im, augment=augment, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Обработка предсказаний
        for i, det in enumerate(pred):  # для каждого изображения
            if len(det):
                # Сохранение результатов
                for *xyxy, conf, cls in reversed(det):
                    # Здесь можно добавить код для обработки результатов, например:
                    # Сохранение распознанного номера в файл
                    print(f'Распознанный номер: {cls} с уверенностью {conf}')

if __name__ == "__main__":
    run()  # Запуск функции
