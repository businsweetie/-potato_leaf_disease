import config
import numpy as np
import gradio as gr
from utils import classification_image, segmentation_image
from ultralytics import YOLO

model_classification_name = gr.Radio([config.CLF_MODEL_LIST[0], config.CLF_MODEL_LIST[1]],
                                    value=config.CLF_MODEL_LIST[0],
                                    label="Модель",
                                    info="Выберите модель для определения уровня заболевания картофеля")
inputs_image = gr.Image(type="filepath", label="Input Image")
outputs_image = gr.Image(type="numpy", label="Output Image")

# Create Gradio Interface for Image Inference
interface_classification_image = gr.Interface(
    fn=classification_image,
    inputs=[inputs_image, model_classification_name],
    outputs=outputs_image,
    title=config.TASK_LIST[0],
    description="Загрузить ваше фото и выберите модель для определения уровня заболевания картофеля",
    examples=[["examples/thumb.jpg"], ["examples/thumb1.jpg"]],
    cache_examples=False,
)

model_segmentation_name = gr.Radio([config.SEG_MODEL_LIST[0], config.SEG_MODEL_LIST[1]],
                                    value=config.SEG_MODEL_LIST[0],
                                    label="Модель",
                                    info="Выберите модель для определения очагов заболевания")

inputs_segm_image = gr.Image(type="filepath", label="Input Image")
outputs_segm_image = gr.Image(type="numpy", label="Output Image")
conf_slider = gr.Slider(0, 1, value=0.5, step=0.05, label='Порог вероятности')
iou_slider = gr.Slider(0, 1, value=0.7, step=0.1, label='Порог IOU')

#Create Gradio Interface for Image Inference
interface_segmentation_image = gr.Interface(
    fn=segmentation_image,
    inputs=[inputs_segm_image, model_segmentation_name, conf_slider, iou_slider],
    outputs=outputs_segm_image,
    title=config.TASK_LIST[1],
    description="Загрузить ваше фото и выберите модель для определения очагов заболевания",
    examples=[["examples/thumb.jpg"], ["examples/thumb1.jpg"]],
    cache_examples=False,
)

# =================== ГЛАВНЫЙ ИНТЕРФЕЙС ================================

interface = gr.TabbedInterface(
    [interface_classification_image, interface_segmentation_image],
    tab_names=[config.TASK_LIST[0], config.TASK_LIST[1]],
    css='.gradio-container {width: 70% !important}',
    )

# запуск приложения
interface.queue().launch(share=True)