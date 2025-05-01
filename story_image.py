import os
import re
from typing import List, Tuple

# Сторонние библиотеки
import cv2
import lpips
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Transformers и обработка текста
import clip
import tiktoken
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
)

# Diffusers и модели генерации
from diffusers import (
    AutoPipelineForImage2Image,
    DDIMScheduler,
    LCMScheduler,
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
)
from diffusers.utils import load_image, make_image_grid

# Torch и нейросетевые модули
from torch import nn
from torch.nn import functional as F
from torchvision import models

def critic(prompt,tokenizer,model):
  
    base_prompt = f'''You get info about the story: Actions, Location, Characters
                    If everything looks correct, answer: YES, else NO
                    ANSWER ONLY YES OR NO
                    INFO:{prompt}'''
    messages = [
        {"role": "system", "content": "You're a critic. You're best in it"},
        {"role": "user", "content": base_prompt}
    ]

    # Применение шаблона чата
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Подготовка входных данных для модели
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Генерация ответа
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=64
    )

    # Извлечение сгенерированного текста
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def story_deepthink(prompt,tokenizer,model):
  
    base_prompt = f'''You've got the text of the story. You should think about it
                    You should in details describe the location and characters of this story
                    And describe the actions of the story.
                    Do it maximum illustrative
                    Story:{prompt}'''
    messages = [
        {"role": "system", "content": "You're a thinker. You're best in it"},
        {"role": "user", "content": base_prompt}
    ]

    # Применение шаблона чата
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Подготовка входных данных для модели
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Генерация ответа
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=64
    )

    # Извлечение сгенерированного текста
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def location_desсriber(prompt, location, tokenizer, model):
  
    base_prompt = f'''You've got the text of the story.
                    You should describe the {location} in this story this story
                    You can imagine how it looks like
                    Do it briefly
                    Story:{prompt}'''
    messages = [
        {"role": "system", "content": "You're a painter. You're best in it"},
        {"role": "user", "content": base_prompt}
    ]

    # Применение шаблона чата
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Подготовка входных данных для модели
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Генерация ответа
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=64
    )

    # Извлечение сгенерированного текста
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def location_finder(prompt,tokenizer,model):
  
    base_prompt = f'''You've got the text of the story.
                    Write a locations in which the actions in the story take place.
                    Be brief as possible and write locations with commas
                    WRITE ONLY LOCATIONS DO NOT WRITE EXTRA WORDS
                    STORY:{prompt}'''
    messages = [
        {"role": "system", "content": "You're a painter. You're best in it"},
        {"role": "user", "content": base_prompt}
    ]

    # Применение шаблона чата
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Подготовка входных данных для модели
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Генерация ответа
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=64
    )

    # Извлечение сгенерированного текста
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def location_matcher(prompt, location,tokenizer, model):
  
    base_prompt = f'''You've got the action and locations.
                    Write a location in which this actions may take place.
                    
                    ACTION:{prompt}
                    LOCATIONS:{location}
                    CHOOSE ONLY ONE LOCATION FROM A LIST DO NOT WRITE OTHER WORDS
                    USE WORDS FROM A LIST DO NOT CHANGE THEM
                    '''
    messages = [
        {"role": "system", "content": "You're a writer. You're best in it"},
        {"role": "user", "content": base_prompt}
    ]

    # Применение шаблона чата
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Подготовка входных данных для модели
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Генерация ответа
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=64
    )

    # Извлечение сгенерированного текста
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def char_desсriber(prompt,tokenizer, model):
  
    base_prompt = f'''You've got the text of the story.
                    You should list the characters
                    Do not use names, descrivbe them in general according their roles
                    Do not use extra words and symbols
                    Story:{prompt}'''
    messages = [
        {"role": "system", "content": "You're a writer. You're best in it"},
        {"role": "user", "content": base_prompt}
    ]

    # Применение шаблона чата
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Подготовка входных данных для модели
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Генерация ответа
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=64
    )

    # Извлечение сгенерированного текста
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def describe_story(prompt,tokenizer, model):
  
    base_prompt = f'''You get a story to input.
                      Your task is to write the description for the Characters and the Location.
                      You can imagine them
                      Describe it briefly in details
                      WRITE ONLY DESCRIPTIONS!!!
                      Text to be processed:{prompt}'''
    messages = [
        {"role": "system", "content": "You're a professional writer. You're best in it"},
        {"role": "user", "content": base_prompt}
    ]

    # Применение шаблона чата
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Подготовка входных данных для модели
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Генерация ответа
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=70
    )

    # Извлечение сгенерированного текста
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def generate_response(prompt,tokenizer, model, parts = 3):
  
    base_prompt = f'''  You need to divide the text ONLY into {parts} actions and describe them, ONLY {parts}, NOT MORE!!!!
                        Seperate each main ACTION and begin with token '[ACTION]'
                        Do not use any characters except letters, numbers, and the dot and comma sign.
                        DO NOT USE OTHER TOKENS!!!!
                        THERE ARE MUST BE ONLY {parts} PARTS!!!!
                        Text to be processed:{prompt}'''
    messages = [
        {"role": "system", "content": "You're a professional text writer. You're best in it"},
        {"role": "user", "content": base_prompt}
    ]

    # Применение шаблона чата
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Подготовка входных данных для модели
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Генерация ответа
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=64
    )

    # Извлечение сгенерированного текста
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def clean_text(text):
    # Регулярное выражение для оставления только букв, цифр, точек и запятых
    cleaned_text = re.sub(r"[^][a-zA-Z0-9 .,]", "", text)
    return cleaned_text

def merge_actions(actions, num=3):
    if len(actions) <= num:
        return actions
    else:
        # Если частей больше 3, объединяем их равномерно
        step = len(actions) // num  # Определяем шаг для объединения
        merged_actions = [
            " ".join(actions[i:i + step]) for i in range(0, len(actions), step)
        ]
        # Если после объединения осталось больше 3 частей, объединяем последние
        if len(merged_actions) > num:
            merged_actions[2] += " " + " ".join(merged_actions[num:])
            merged_actions = merged_actions[:num]
        return merged_actions

def calculate_clip_score(image: str, text: str, model_name: str = "ViT-B/32") -> float:
    """
    Вычисляет CLIP Score — меру соответствия между изображением и текстом.
    
    Параметры:
        image_path (str): Путь к изображению.
        text (str): Текст для сравнения.
        model_name (str): Название модели CLIP (по умолчанию "ViT-B/32").
    
    Возвращает:
        float: Значение CLIP Score (косинусное сходство между эмбеддингами).
    """
    # Загрузка модели и предобработки
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    
    # Загрузка и предобработка изображения
    image = preprocess(image).unsqueeze(0).to(device)
    
    # Токенизация текста
    text_tokens = clip.tokenize([text]).to(device)
    
    # Вычисление эмбеддингов
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
    
    # Косинусное сходство
    clip_score = torch.cosine_similarity(image_features, text_features).item()
    
    return clip_score

def calculate_lpips(image_path1: str, image_path2: str, net_type: str = 'alex') -> float:
    """
    Вычисляет метрику LPIPS между двумя изображениями.

    Параметры:
        image_path1 (str): Путь к первому изображению.
        image_path2 (str): Путь ко второму изображению.
        net_type (str): Модель для вычисления ('alex', 'vgg', 'squeeze').

    Возвращает:
        float: Значение LPIPS (чем меньше, тем более схожи изображения).
    """
    # Инициализация LPIPS
    loss_fn = lpips.LPIPS(net=net_type).eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn.to(device)

    def load_image(path):
        img = path#Image.open(path).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform(img).unsqueeze(0).to(device)

    img1 = load_image(image_path1)
    img2 = load_image(image_path2)

    # Расчет LPIPS
    with torch.no_grad():
        distance = loss_fn(img1, img2).item()

    return distance

def parse_characters(input_str):
    # Удаляем заголовок "Characters:" (регистронезависимо)
    input_str = re.sub(r'^.*characters:.*$', '', input_str, flags=re.IGNORECASE | re.MULTILINE)
    
    # Находим все элементы списка
    entries = re.findall(r'^\s*-\s*(.*?)\s*$', input_str, flags=re.MULTILINE)
    
    # Очистка и форматирование каждого элемента
    cleaned_entries = []
    for entry in entries:
        # Удаляем лишние символы, кроме букв, цифр, пробелов и скобок
        entry = re.sub(r'[^\w\s()]', '', entry)
        # Убираем лишние пробелы
        entry = re.sub(r'\s+', ' ', entry).strip()
        # Приводим к Title Case (первые буквы слов заглавные)
        entry = entry.title()
        cleaned_entries.append(entry)
    
    return ', '.join(cleaned_entries)

def parse_actions(input_str):
    # Разделяем по [ACTION] в любом регистре
    actions = re.split(r'\[ACTION\]', input_str, flags=re.IGNORECASE)
    
    # Очищаем от пустых строк и лишних пробелов
    cleaned_actions = [action.strip() for action in actions if action.strip()]
    
    return cleaned_actions

def create_story_report(story_ls: list, characters: list, location_ls: list) -> str:
    """
    Объединяет списки сюжетов, персонажей и локаций в форматированный текстовый отчет.
    
    Параметры:
        story_ls: Список сюжетных событий
        characters: Список персонажей (с атрибутами)
        location_ls: Список локаций
    
    Возвращает:
        str: Форматированный текстовый отчет
    """
    # Заголовок и сюжет
    report = "📖 СЮЖЕТНЫЕ СОБЫТИЯ:\n"
    report += "\n".join(f"→ {i+1}. {story}" for i, story in enumerate(story_ls))
    
    # Персонажи
    report += "\n\n🎭 ПЕРСОНАЖИ:\n"
    if isinstance(characters, dict):  # Если characters - словарь
        report += "\n".join(f"• {name}: {desc}" for name, desc in characters.items())
    else:  # Если characters - список
        report += "\n".join(f"• {char}" for char in characters)
    
    # Локации
    report += "\n\n🌍 ЛОКАЦИИ:\n"
    unique_locations = list(set(location_ls))  # Уникальные локации
    report += "\n".join(f"📍 {loc}" for loc in unique_locations)
    
    
    return report

def llm_agents(story: str,tokenizer, model) -> Tuple[List[str], dict, List[str]]:
    """Генерация согласованных story_ls и location_ls с контролем длины токенов."""
    enc = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(text: str) -> int:
        return len(enc.encode(text))

    # 1. Получение персонажей (без ограничений токенов)
    characters = parse_characters(char_desсriber(story,tokenizer, model))
    
    # 2. Генерация story_ls (макс 3 попытки)
    story_ls = []
    for _ in range(3):
        story_ls = merge_actions(parse_actions(generate_response(story,tokenizer, model)))
        if all(count_tokens(item) < 77 for item in story_ls):
            break
    else:
        story_ls = [s[:70] + "..." for s in story_ls]  # Принудительная обрезка
    
    # 3. Генерация location_ls (макс 3 попытки)
    locations_raw = location_finder(story,tokenizer, model)
    location_ls = []
    
    for attempt in range(3):
        location_ls = []
        for story_item in story_ls:
            loc = location_matcher(story_item, locations_raw,tokenizer, model)
            location_ls.append(loc[:70] + "..." if count_tokens(loc) >= 77 else loc)
        
        # Корректировка длины
        if len(location_ls) != len(story_ls):
            diff = len(story_ls) - len(location_ls)
            filler = location_ls[0] if len(set(location_ls)) == len(location_ls) else location_ls[-1]
            location_ls.extend([filler] * diff)
        
        # Проверка успешности
        if (len(location_ls) == len(story_ls) and 
            all(count_tokens(loc) < 77 for loc in location_ls)):
            if 'yes' in critic(create_story_report(story_ls, characters, location_ls),tokenizer, model).lower():
                break
    
    # Финальные гарантии
    story_ls = [s if count_tokens(s) < 77 else s[:70] + "..." for s in story_ls]
    location_ls = location_ls[:len(story_ls)]
    location_ls = [loc if count_tokens(loc) < 77 else loc[:70] + "..." 
                  for loc in location_ls]
    
    return story_ls, characters, location_ls

def generate_best_character_image(pipeline, characters: str, generator, empty_image, show_results: bool = False) -> Image.Image:
    """
    Генерирует 3 изображения персонажа и возвращает лучшее по CLIP Score.
    
    Параметры:
        pipeline: Пайплайн для генерации изображений
        characters (str): Описание персонажа
        generator: Генератор для воспроизводимости
        empty_image: Изображение для IP-адаптера
        show_results (bool): Показывать все варианты (по умолчанию False)
    
    Возвращает:
        Image.Image: Лучшее изображение по CLIP Score
    """
    pipeline.set_ip_adapter_scale(0)
    
    # Генерация 3 вариантов
    generated_images = []
    clip_scores = []
    
    for i in range(3):
        char_img = pipeline(
            prompt=f'Illustration {characters}',
            ip_adapter_image=empty_image,
            negative_prompt="lowres, bad anatomy, worst quality, low quality",
            num_inference_steps=50,
            generator=generator,
        ).images[0]
        
        score = calculate_clip_score(char_img, characters)
        generated_images.append(char_img)
        clip_scores.append(score)
        
        if show_results:
            plt.figure(figsize=(5, 5))
            plt.imshow(char_img)
            plt.title(f"Вариант {i+1} - CLIP: {score:.4f}")
            plt.axis('off')
            plt.show()
    
    # Выбор лучшего изображения
    best_idx = np.argmax(clip_scores)
    best_image = generated_images[best_idx]
    best_score = clip_scores[best_idx]
    
    print(f"Лучший CLIP Score: {best_score:.4f} (вариант {best_idx+1})")
    
    if show_results:
        plt.figure(figsize=(6, 6))
        plt.imshow(best_image)
        plt.title(f"Лучший результат - CLIP: {best_score:.4f}")
        plt.axis('off')
        plt.show()
    
    return best_image

def generate_backgrounds_with_quality_check(
    pipeline,
    characters: str,
    location_ls: list,
    char_image: Image.Image,
    generator,
    negative_prompt: str = "lowres, bad anatomy, worst quality, low quality",
    min_lpips_quality: float = 0.5,
    max_attempts: int = 3
) -> dict:
    """
    Генерирует фоны для локаций с проверкой качества через LPIPS.
    
    Параметры:
        pipeline: Пайплайн для генерации изображений
        characters (str): Описание персонажа
        location_ls: Список уникальных локаций
        char_image: Изображение персонажа для сравнения
        generator: Генератор для воспроизводимости
        negative_prompt: Негативный промпт
        min_lpips_quality: Минимальное требуемое качество (1 - LPIPS)
        max_attempts: Максимальное число попыток генерации
    
    Возвращает:
        dict: Словарь {локация: Image} с гарантированным качеством
    """
    pipeline.set_ip_adapter_scale(0.2)
    background_dict = {}
    
    for loc in list(set(location_ls)):
        for attempt in range(max_attempts):
            # Генерация фона
            background = pipeline(
                prompt=f'High-quality illustration of {characters} in {loc}',
                ip_adapter_image=char_image,
                negative_prompt=negative_prompt,
                num_inference_steps=50,
                generator=generator,
            ).images[0]
            
            # Проверка качества
            lpips_score = calculate_lpips(char_image, background)
            quality = 1 - lpips_score
            
            if quality >= min_lpips_quality:
                background_dict[loc] = background
                print(f"✓ Локация '{loc}': качество {quality:.4f} (попытка {attempt+1})")
                break
                
            print(f"× Локация '{loc}': качество {quality:.4f} < {min_lpips_quality} (попытка {attempt+1})")
            
            if attempt == max_attempts - 1:
                background_dict[loc] = background  # Сохраняем лучший из неудачных вариантов
                print(f"⚠ Для локации '{loc}' достигнут максимум попыток. Лучшее качество: {quality:.4f}")
    
    # Визуализация результатов
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(char_image)
    plt.title("Персонаж")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    first_loc = next(iter(background_dict))
    plt.imshow(background_dict[first_loc])
    plt.title(f"Фон: {first_loc}\nLPIPS качество: {1 - calculate_lpips(char_image, background_dict[first_loc]):.4f}")
    plt.axis('off')
    
    plt.show()
    
    return background_dict

def generate_best_story_frames(
    pipeline,
    story_ls: list,
    location_ls: list,
    background_dict: dict,
    characters: str,
    generator,
    negative_prompt: str = "lowres, bad anatomy, worst quality, low quality",
    num_variants: int = 3,
    min_lpips: float = 0.5,
    display: bool = True
) -> list:
    """
    Генерирует лучшие кадры для историй с двойной проверкой качества (CLIP + LPIPS).
    
    Параметры:
        pipeline: Пайплайн генерации
        story_ls: Список историй
        location_ls: Список соответствующих локаций
        background_dict: Словарь фонов {локация: Image}
        characters: Описание персонажей
        generator: Генератор для воспроизводимости
        negative_prompt: Негативный промпт
        num_variants: Количество вариантов на историю
        min_lpips: Минимальное требование к LPIPS (1 - расстояние)
        display: Показывать результаты
    
    Возвращает:
        list: Лучшие изображения для каждой истории
    """
    pipeline.set_ip_adapter_scale(0.2)
    best_frames = []
    best_clips = []
    best_lpips = []
    
    for i, (story, loc) in enumerate(zip(story_ls, location_ls)):
        top_candidates = []
        
        # Генерация вариантов
        for _ in range(num_variants):
            img = pipeline(
                prompt=f'Illustration: {story} with {characters}',
                ip_adapter_image=background_dict.get(loc),
                negative_prompt=negative_prompt,
                num_inference_steps=50,
                generator=generator,
            ).images[0]
            
            # Вычисление метрик
            clip_score = calculate_clip_score(img, story)
            lpips_dist = calculate_lpips(background_dict[loc], img)
            lpips_quality = 1 - lpips_dist
            
            top_candidates.append({
                'image': img,
                'clip': clip_score,
                'lpips': lpips_quality,
                'combined_score': clip_score * (lpips_quality if lpips_quality >= min_lpips else 0)
            })
        
        # Выбор лучшего по комбинированному score
        top_candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        best = top_candidates[0]
        best_frames.append(best['image'])
        best_clips.append(best['clip'])
        best_lpips.append(best['lpips'])
        if display:
            print(f"История {i+1}:")
            print(f"  CLIP: {best['clip']:.4f} | LPIPS: {best['lpips']:.4f}")
            print(f"  Текст: {story[:50]}...")
    
    # Визуализация
    if display:
        plt.figure(figsize=(20, 5))
        for idx, img in enumerate(best_frames):
            plt.subplot(1, len(best_frames), idx + 1)
            plt.imshow(img)
            plt.title(f"Кадр {idx+1}\n{story_ls[idx][:30]}...")
            plt.axis("off")
        plt.tight_layout()
        plt.show()
    
    return best_frames, best_clips, best_lpips

def save_images(image_list, num_act, output_dir="output_images", prefix="frame"):
    """
    Сохраняет список изображений в указанную директорию
    
    :param image_list: Список объектов PIL.Image
    :param output_dir: Папка для сохранения (по умолчанию "output_images")
    :param prefix: Префикс имен файлов (по умолчанию "frame")
    """
    # Создаем папку, если ее нет
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img in enumerate(image_list):
        # Формируем имя файла: frame_1.jpg, frame_2.jpg и т.д.
        filename = f"{prefix}_{str(num_act)}_{i+1}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        # Сохраняем изображение в формате JPEG
        img.save(filepath, "JPEG", quality=95)
        print(f"Сохранено: {filepath}")

def gen_img(text,
           use_lora = True):

    ## LLM
    model_name = "Qwen/Qwen2.5-1.5B-Instruct" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda").to("cuda")
    
    ## Stable Diffusion
    pipeline = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to("cuda")
    
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    
    pipeline.set_ip_adapter_scale(0.2)

    if use_lora:
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        pipeline.load_lora_weights("/kaggle/input/load-lora-weights-pipe-unet-load-attn-procs/out/")
        pipeline.fuse_lora()
    
    # Создание пустого изображения
    #empty_image = Image.new("RGB", (512, 512), (255, 255, 255))  # Белый фон, размер 512x512
    width, height = 512, 512
    
    # Создаем массив случайных значений (шум) в диапазоне [0, 255]
    empty_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    generator = torch.Generator(device="cpu").manual_seed(26)

    story_ls, characters, location_ls = llm_agents(stories[i],tokenizer, model)
    
    story_report = create_story_report(
                                    story_ls=story_ls,
                                    characters=characters,
                                    location_ls=location_ls)

    best_char_image = generate_best_character_image(
                                                    pipeline=pipeline,
                                                    characters=characters,
                                                    generator=generator,
                                                    empty_image=empty_image,
                                                    show_results=True)

    background_dict = generate_backgrounds_with_quality_check(
                                                            pipeline=pipeline,
                                                            characters=characters,
                                                            location_ls=location_ls,
                                                            char_image=best_char_image,
                                                            generator=generator,
                                                            min_lpips_quality=0.5)

    best_frames, best_clips, best_lpips = generate_best_story_frames(
                                                                    pipeline=pipeline,
                                                                    story_ls=story_ls,
                                                                    location_ls=location_ls,
                                                                    background_dict=background_dict,
                                                                    characters=characters,
                                                                    generator=generator)
    
    return best_frames, best_clips, best_lpips
        