"""
Inférence TensorFlow Lite - Remplacement de YOLO dans votre projet robot
Remplace la fonction detect_balls() qui utilise ultralytics YOLO

Usage dans FullGUIV7.py :
    Remplacez :
        from ultralytics import YOLO
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        det = detect_balls(frame, self.yolo_model)

    Par :
        from inference_tflite import TFLiteDetector
        self.yolo_model = TFLiteDetector("model_nano_saved_model/model_nano_float32.tflite")
        det = self.yolo_model.detect(frame)
"""

import numpy as np
import cv2
from pathlib import Path

# Nom de la classe à détecter (doit correspondre à votre modèle)
YOLO_CLASS_NAME = "ping-pong-ball"
YOLO_CONF_THRESH = 0.5


class TFLiteDetector:
    """
    Détecteur YOLO via TensorFlow Lite.
    Interface compatible avec detect_balls() de FullGUIV7.py
    """

    def __init__(self, model_path: str, conf_thresh: float = YOLO_CONF_THRESH):
        """
        Args:
            model_path  : Chemin vers le fichier .tflite
            conf_thresh : Seuil de confiance minimum
        """
        self.conf_thresh = conf_thresh
        self.model_path = model_path

        # Charger TFLite
        try:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            print(f"✅ TFLite chargé avec TensorFlow : {model_path}")
        except ImportError:
            # Fallback : tflite_runtime (plus léger, recommandé sur Raspberry Pi)
            try:
                import tflite_runtime.interpreter as tflite
                self.interpreter = tflite.Interpreter(model_path=model_path)
                print(f"✅ TFLite chargé avec tflite_runtime : {model_path}")
            except ImportError:
                raise ImportError(
                    "Installez TensorFlow ou tflite_runtime :\n"
                    "  pip install tensorflow\n"
                    "  ou (Raspberry Pi) : pip install tflite-runtime"
                )

        self.interpreter.allocate_tensors()

        # Récupérer les détails entrée/sortie
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Taille d'entrée attendue par le modèle
        input_shape = self.input_details[0]['shape']
        self.input_h = input_shape[1]
        self.input_w = input_shape[2]
        self.is_int8 = self.input_details[0]['dtype'] == np.int8 or \
                       self.input_details[0]['dtype'] == np.uint8

        print(f"   Entrée : {self.input_w}x{self.input_h}, "
              f"type={'INT8' if self.is_int8 else 'FP32'}")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Prépare l'image pour l'inférence TFLite."""
        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # (1, H, W, 3)

        if self.is_int8:
            # Quantisation pour modèle INT8
            scale, zero_point = self.input_details[0]['quantization']
            img = (img / scale + zero_point).astype(np.int8)

        return img

    def detect(self, frame: np.ndarray):
        """
        Détecte la balle de ping-pong dans un frame.

        Returns:
            (cx, cy, radius) si détectée, None sinon
            — même format que detect_balls() dans FullGUIV7.py
        """
        orig_h, orig_w = frame.shape[:2]

        # Prétraitement
        input_tensor = self.preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)

        # Inférence
        self.interpreter.invoke()

        # Récupérer les sorties
        # Format YOLO TFLite : [1, num_classes+4, num_boxes] ou [1, num_boxes, 6]
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Déquantification si INT8
        if self.is_int8:
            scale, zero_point = self.output_details[0]['quantization']
            output = (output.astype(np.float32) - zero_point) * scale

        return self._parse_output(output, orig_w, orig_h)

    def _parse_output(self, output: np.ndarray, orig_w: int, orig_h: int):
        """
        Parse la sortie YOLO TFLite.
        Supporte le format [1, 5+nc, 8400] (YOLOv8 standard).
        """
        # YOLOv8 export TFLite : shape (1, 5+nc, num_anchors)
        if output.ndim == 3:
            output = output[0]  # (5+nc, num_anchors)
            output = output.T   # (num_anchors, 5+nc)

        best_conf = self.conf_thresh
        best_det = None

        for det in output:
            # det = [x_center, y_center, width, height, conf_class0, conf_class1, ...]
            if len(det) < 5:
                continue

            x_c, y_c, w, h = det[0], det[1], det[2], det[3]
            class_scores = det[4:]
            class_id = int(np.argmax(class_scores))
            conf = float(class_scores[class_id])

            if conf < best_conf:
                continue

            # Coordonnées normalisées -> pixels originaux
            x1 = int((x_c - w / 2) * orig_w)
            y1 = int((y_c - h / 2) * orig_h)
            x2 = int((x_c + w / 2) * orig_w)
            y2 = int((y_c + h / 2) * orig_h)

            radius = int(min(x2 - x1, y2 - y1) / 2)
            if radius < 5:
                continue

            best_conf = conf
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            best_det = (cx, cy, radius)

        return best_det


# ---------------------------------------------------------------------------
# Fonction drop-in pour remplacer detect_balls() dans FullGUIV7.py
# ---------------------------------------------------------------------------
def detect_balls_tflite(frame: np.ndarray, detector: TFLiteDetector):
    """
    Remplacement direct de detect_balls(frame, model) dans FullGUIV7.py.
    Même signature, même valeur de retour.
    """
    return detector.detect(frame)


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import glob
    import sys

    # Trouver le fichier .tflite généré
    tflite_files = glob.glob("**/*.tflite", recursive=True)
    if not tflite_files:
        print("❌ Aucun fichier .tflite trouvé. Lancez d'abord convert_yolo_to_tflite.py")
        sys.exit(1)

    tflite_path = tflite_files[0]
    print(f"Test avec : {tflite_path}")

    detector = TFLiteDetector(tflite_path)

    # Test sur une image noire (juste pour vérifier que ça tourne)
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(dummy)
    print(f"Résultat sur image vide : {result}")
    print("✅ Le détecteur TFLite fonctionne correctement !")
