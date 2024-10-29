import logging
import requests
from telegram import InputMediaPhoto, Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters
from telegram.ext import CallbackContext
from io import BytesIO
import os
import re
import tempfile
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Ваши ключи и токены
BING_SUBSCRIPTION_KEY = 'fe449a4a0b6a49b8a9192610ae26ebf0'
BING_ENDPOINT = 'https://api.bing.microsoft.com/v7.0/images/search'
COMPUTER_VISION_SUBSCRIPTION_KEY = 'bd87d569695c40fbbb88eb384730c594'
COMPUTER_VISION_ENDPOINT = 'https://eastus.api.cognitive.microsoft.com/'
TELEGRAM_TOKEN = '7561185124:AAFH4439bQ3Cme8ZK_KNSYkV2Xn-xomNcc8'

# Настройка Azure Computer Vision
vision_client = ComputerVisionClient(COMPUTER_VISION_ENDPOINT, CognitiveServicesCredentials(COMPUTER_VISION_SUBSCRIPTION_KEY))

# Функция для очистки распознанного текста
def clean_extracted_text(extracted_text):
    cleaned_text = extracted_text.strip()
    cleaned_text = re.sub(r'^[KК]Z\s?', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', '', cleaned_text)
    return cleaned_text

# Функция для распознавания текста на изображении
def extract_text_from_image(image_path):
    with open(image_path, "rb") as image_stream:
        ocr_result = vision_client.recognize_printed_text_in_stream(image_stream)
        
    extracted_text = ""
    for region in ocr_result.regions:
        for line in region.lines:
            for word in line.words:
                extracted_text += word.text + " "
    
    logger.info(f"Extracted text: {extracted_text.strip()}")
    
    cleaned_text = clean_extracted_text(extracted_text.strip())
    
    logger.info(f"Cleaned text: {cleaned_text}")
    
    return cleaned_text

# Функция для скрейпинга Instagram
def search_images_in_instagram_by_text(query):
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    logger.info(f"Searching Instagram for query: {query}")
    
    url = f"https://www.instagram.com/explore/tags/{query}/"
    driver.get(url)
    
    time.sleep(5)  # Ожидание загрузки страницы
    
    images = driver.find_elements(By.TAG_NAME, 'img')
    
    image_urls = []
    for image in images:
        image_url = image.get_attribute('src')
        if image_url:
            logger.info(f"Found image: {image_url}")
            image_urls.append(image_url)

    logger.info(f"Total images found: {len(image_urls)}")
    
    driver.quit()

    return image_urls[:5]  # Возвращаем только первые 5 изображений

# Функция для обработки изображений, отправленных пользователем
async def handle_image(update: Update, context: CallbackContext):
    try:
        logger.info(f"Received an image from user {update.message.from_user.id}")
        photo_file = await update.message.photo[-1].get_file()
        image_bytes = BytesIO()
        await photo_file.download_to_memory(out=image_bytes)
        image_bytes.seek(0)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file.write(image_bytes.getvalue())
            temp_file_name = temp_file.name

        extracted_text = extract_text_from_image(temp_file_name)
        os.remove(temp_file_name)

        if extracted_text:
            await update.message.reply_text(f'Распознанный номер: {extracted_text}')

            pattern = r'\d{3}[АВЕКМНОРСТУХ]{3}\d{2}'  # Регулярное выражение для номеров формата 327ААК15

            if re.match(pattern, extracted_text):
                similar_images = search_images_in_instagram_by_text(extracted_text)

                if similar_images:
                    logger.info(f"Found {len(similar_images)} similar images, sending to user")
                    media = [InputMediaPhoto(img_url) for img_url in similar_images]
                    await update.message.reply_media_group(media)
                else:
                    await update.message.reply_text('Похожие изображения не найдены в Instagram.')
            else:
                await update.message.reply_text(f'Распознанный текст "{extracted_text}" не похож на номер автомобиля.')
        else:
            await update.message.reply_text('Не удалось распознать номер на изображении.')
    
    except Exception as e:
        logger.error(f"An error occurred while handling the image: {e}")
        await update.message.reply_text('Произошла ошибка при обработке изображения.')

# Команда /start
async def start(update: Update, context: CallbackContext):
    logger.info(f"User {update.message.from_user.id} started the bot")
    await update.message.reply_text('Привет! Отправь изображение с номером автомобиля, и я найду похожие изображения!')

# Запуск бота
def main():
    logger.info("Starting the bot")
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    application.run_polling()

if __name__ == '__main__':
    main()
