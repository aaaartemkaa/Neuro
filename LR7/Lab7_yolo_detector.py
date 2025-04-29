import os
import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import yaml

def setup_environment():
    "Проверка и настройка окружения"
    print("1. Проверка окружения...")
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("backup"):
        os.makedirs("backup")
    print(" Окружение готово! :)")

def verify_dataset_structure():
    "Проверка структуры датасета"
    print("\n2. Проверка структуры датасета:")
    
    required_files = {
        "data/obj.yaml": "YAML конфигурационный файл",
        "data/train.txt": "список изображений"
    }
    
    for file, desc in required_files.items():
        if not os.path.exists(file):
            raise FileNotFoundError(f" {desc} '{file}' не найден!")
        print(f" {desc} существует")
        
    # Проверка наличия изображений
    with open("data/train.txt", "r") as f:
        images = [line.strip() for line in f.readlines()]
        missing_images = [img for img in images if not os.path.exists(img)]
        if missing_images:
            print("\nПредупреждение: отсутствуют следующие изображения:")
            for img in missing_images[:3]:
                print(f"- {img}")
            if len(missing_images) > 3:
                print(f"...и еще {len(missing_images)-3} изображений")
    print(" Структура датасета проверена :)")
    
def create_yaml_config():
    "Создание YAML конфига для YOLOv8"
    config = {
        'path': os.path.abspath("data"),
        'train': 'train.txt',             
        'val': 'train.txt',
        'names': {                        
            0: 'I',
            1: 'Nikita',
            2: 'Evgenyi',
            3: 'Ivan',
            4: 'Others'
        }
    }
    with open("data/obj.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(" YAML конфиг создан")

def load_model():
    "Загрузка предобученной модели"
    print("\n3. Загрузка модели YOLOv8...")
    try:
        # Проверяем наличие файла модели
        if not os.path.exists("yolov8s.pt"):
            print("Файл модели не найден, начинаем загрузку...")
            
        model = YOLO("yolov8s.pt")
        print(" Модель успешно загружена :)")
        return model
    except Exception as e:
        print(f" Ошибка загрузки модели: {str(e)}")
        return None

def train_model(model):
    "Обучение модели"
    if model is None:
        print(" Модель не загружена, обучение невозможно")
        return None
    
    print("\n4. Настройка параметров обучения:")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"- Используемое устройство: {device}")
    
    # Считаем количество изображений
    with open("data/train.txt", "r") as f:
        num_images = len(f.readlines())
    print(f"- Изображений для обучения: {num_images}")

    # Параметры обучения
    train_args = {
        "data": os.path.abspath("data/obj.yaml"),
        "epochs": 60,
        "batch": 8,
        "imgsz": 640,
        "project": "custom_yolo",
        "name": "exp1",
        "val": True,
        "device": device,
        "exist_ok": True
    }

    print("\n5. Запуск обучения...")
    try:
        results = model.train(**train_args)
        print("\n Обучение успешно завершено!")
        return model
    except Exception as e:
        print(f"\n Ошибка при обучении: {str(e)}")
        return None

def test_model(model, image_path="data/obj_train_data/photo_2025-04-23_12-40-13.jpg"):
    "Тестирование модели"
    if model is None:
        print(" Модель не обучена, тестирование невозможно")
        return
    test_image = image_path
    print(f"\n6. Тестирование на изображении: {test_image}")
    
    if not os.path.exists(test_image):
        print(f" Изображение {test_image} не найдено")
        return
    
    try:
        results = model(test_image)
        result = results[0]
        
        # Визуализация
        plt.figure(figsize=(12, 8))
        plt.imshow(result.plot()[:, :, ::-1])
        plt.axis('off')
        plt.title("Результаты детекции")
        plt.show()
        
        print("\nОбнаруженные объекты:")
        for i, box in enumerate(result.boxes, 1):
            class_id = int(box.cls)
            class_name = result.names[class_id]
            confidence = float(box.conf)
            bbox = [round(x, 2) for x in box.xyxy[0].tolist()]
            print(f"{i}. {class_name} (уверенность: {confidence:.2f}, координаты: {bbox})")
            
    except Exception as e:
        print(f" Ошибка при тестировании: {str(e)}")
        
def process_video(model, video_path="videos/test_video.mp4"):
    "Обработка видеофайла"
    if model is None:
        print(" Модель не обучена, обработка видео невозможна")
        return
        
    print(f"\n7. Обработка видео: {video_path}")
    
    if not os.path.exists(video_path):
        print(f" Видеофайл {video_path} не найден")
        return
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(" Ошибка открытия видеофайла")
            return
        
        # Получаем параметры видео
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Создаем VideoWriter для сохранения результата
        output_path = "videos/output_video.mp4"
        out = cv2.VideoWriter(output_path, 
                             cv2.VideoWriter_fourcc(*'mp4v'), 
                             fps, 
                             (frame_width, frame_height))

        print("Обработка видео (нажмите 'q' для остановки)...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Детекция объектов
            results = model(frame)
            annotated_frame = results[0].plot()
            
            # Отображение и сохранение кадра
            cv2.imshow("YOLOv8 Detection", annotated_frame)
            out.write(annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Обработка завершена :) Результат сохранен как {output_path}")
        
    except Exception as e:
        print(f" Ошибка при обработке видео: {str(e)}")
if __name__ == "__main__":
    try:
        # 0. Обновление ultralytics
        os.system("pip install --upgrade ultralytics")
        
        # 1. Настройка окружения
        setup_environment()
        
        # 2. Создание YAML конфиг
        create_yaml_config()
        
        # 3. Проверка данных
        verify_dataset_structure()
        
        # 4. Загрузка модели
        model = load_model()
        
        # 5. Обучение
        trained_model = train_model(model)
        
        # 6. Тестирование
        if trained_model:
            test_model(trained_model)
            
        # 7. Обработка видео 
            if os.path.exists("videos/test_video.mp4"):
                process_video(trained_model)
            
    except Exception as e:
        print(f"\n Критическая ошибка: {str(e)}")
    finally:
        print("\nРабота программы завершена =)")