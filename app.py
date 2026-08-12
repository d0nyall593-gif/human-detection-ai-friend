import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import time


def render_page(deepface_models_loaded):
    st.header("Анализ от уеб камера (Пол, Възраст, Емоция - Real-time)")
    st.write("Стартирайте уеб камерата за анализ на лица в реално време.")
    st.warning(
        "Анализът в реално време може да е бавен в зависимост от вашия хардуер.")

    if not deepface_models_loaded:
        st.error("DeepFace моделите не са заредени. Функционалността не е достъпна.")
        st.stop()

    detector_backend_webcam = st.sidebar.selectbox(
        "DeepFace детектор за уеб камера:",
        ('opencv', 'ssd', 'mtcnn', 'retinaface', 'yunet'), index=0, key="webcam_detector_module")

    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False
    if 'cap_webcam' not in st.session_state:
        st.session_state.cap_webcam = None

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("🚀 Старт на камерата", key="start_cam_button_module", disabled=st.session_state.webcam_running):
            st.session_state.webcam_running = True
            # При старт, ако cap_webcam съществува, го освобождаваме, за да се инициализира наново
            if st.session_state.cap_webcam is not None:
                st.session_state.cap_webcam.release()
                st.session_state.cap_webcam = None
            st.rerun()

    with col_stop:
        if st.button("🛑 Стоп на камерата", key="stop_cam_button_module", disabled=not st.session_state.webcam_running):
            st.session_state.webcam_running = False
            st.rerun()  # Презареждаме, за да отразим промяната и да спрем цикъла

    image_placeholder = st.empty()

    if st.session_state.webcam_running:
        if st.session_state.cap_webcam is None:
            st.session_state.cap_webcam = cv2.VideoCapture(0)
            if not st.session_state.cap_webcam.isOpened():
                st.error(
                    "Не може да се отвори уеб камерата. Проверете дали е свързана и не се използва от друго приложение.")
                st.session_state.webcam_running = False
                if st.session_state.cap_webcam is not None:
                    st.session_state.cap_webcam.release()  # Освобождаваме, ако е отворена неуспешно
                st.session_state.cap_webcam = None
                st.rerun()  # Презареждаме, за да се покаже грешката и да се нулира състоянието

        if st.session_state.cap_webcam is not None and st.session_state.cap_webcam.isOpened():
            st.info("Уеб камерата е активна...")
            while st.session_state.webcam_running:  # Цикълът продължава, докато webcam_running е True
                ret, frame_bgr = st.session_state.cap_webcam.read()
                if not ret:
                    st.warning(
                        "Не може да се прочете кадър от уеб камерата. Спирам.")
                    st.session_state.webcam_running = False
                    break

                frame_to_analyze_bgr = frame_bgr.copy()
                frame_display_bgr = frame_bgr.copy()

                try:
                    results = DeepFace.analyze(
                        img_path=frame_to_analyze_bgr,
                        actions=['age', 'gender', 'emotion'],
                        enforce_detection=True,
                        detector_backend=detector_backend_webcam,
                        silent=True
                    )

                    if isinstance(results, list) and len(results) > 0:
                        for face_info in results:
                            if 'region' in face_info and face_info['region']['w'] > 0 and face_info['region']['h'] > 0:
                                region = face_info['region']
                                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                                age = face_info.get('age', "N/A")

                                gender_map = face_info.get('gender', {})
                                if isinstance(gender_map, dict) and gender_map:
                                    dominant_gender = max(
                                        gender_map, key=gender_map.get)
                                else:
                                    dominant_gender = face_info.get(
                                        'dominant_gender', "N/A")

                                emotion_map = face_info.get('emotion', {})
                                if isinstance(emotion_map, dict) and emotion_map:
                                    dominant_emotion = max(
                                        emotion_map, key=emotion_map.get)
                                else:
                                    dominant_emotion = face_info.get(
                                        'dominant_emotion', "N/A")

                                cv2.rectangle(
                                    frame_display_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                text_gender_char = dominant_gender[0].upper(
                                ) if dominant_gender != "N/A" and isinstance(dominant_gender, str) else "N"
                                text = f"A:{age} G:{text_gender_char} E:{dominant_emotion}"
                                cv2.putText(frame_display_bgr, text, (x, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except ValueError:
                    pass  # Няма засечени лица (от enforce_detection=True)
                except Exception as e:
                    # Записваме други грешки в конзолата, но не спираме камерата заради тях
                    print(f"Грешка по време на анализ от уеб камера: {e}")
                    pass

                frame_display_rgb = cv2.cvtColor(
                    frame_display_bgr, cv2.COLOR_BGR2RGB)
                image_placeholder.image(frame_display_rgb, channels="RGB")

                time.sleep(0.01)

                if not st.session_state.webcam_running:  # Проверяваме отново, ако бутон "Стоп" е натиснат
                    break

            # Почистване след края на цикъла или при спиране
            if st.session_state.cap_webcam is not None:
                st.session_state.cap_webcam.release()
                st.session_state.cap_webcam = None
            image_placeholder.empty()
            if not st.session_state.webcam_running:  # Ако е спряна от бутон "Стоп"
                st.info("Уеб камерата е спряна.")
            # Ако webcam_running е False (например заради грешка при четене на кадър),
            # съобщението "Уеб камерата е спряна." ще се покаже при следващото презареждане,
            # ако не е вече показано друго съобщение за грешка.

    # Допълнителна проверка за почистване, ако webcam_running е False, но cap все още съществува
    if not st.session_state.webcam_running:
        if st.session_state.get('cap_webcam') is not None:
            st.session_state.cap_webcam.release()
            st.session_state.cap_webcam = None
        image_placeholder.empty()
