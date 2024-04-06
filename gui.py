import os
import ttkbootstrap as tk
from typing import Optional, Tuple
import PIL
import pyttsx3
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow  # noqa

tf = tensorflow.compat.v1

WIDTH = 1024
HEIGHT = 720


class Predictor:
    def __init__(self, label_file_path: str, graph_file_path: str):
        self.label_lines = [line.rstrip() for line in tf.gfile.GFile(label_file_path)]

        # Unpersists graph from file
        with tf.gfile.FastGFile(graph_file_path, "rb") as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(self.graph_def, name="")

        self.session = tf.Session()
        self.softmax_tensor = self.session.graph.get_tensor_by_name("final_result:0")

    def predict(self, frame):
        predictions = self.session.run(
            self.softmax_tensor, {"DecodeJpeg/contents:0": frame}
        )

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]) :][::-1]  # noqa

        max_score = 0.0
        res = ""
        for node_id in top_k:
            human_string = self.label_lines[node_id]
            score = predictions[0][node_id]
            if score > max_score:
                max_score = score
                res = human_string
        return res, max_score

    def __del__(self):
        self.session.close()


class MainApplication(tk.Window):
    def __init__(
        self,
        title="ttkbootstrap",
        themename="litera",
        iconphoto="",
        size=None,
        position=None,
        minsize=None,
        maxsize=None,
        resizable=None,
        hdpi=True,
        scaling=None,
        transient=None,
        overrideredirect=False,
        alpha=1,
    ):
        super().__init__(
            title,
            themename,
            iconphoto,
            size,
            position,
            minsize,
            maxsize,
            resizable,
            hdpi,
            scaling,
            transient,
            overrideredirect,
            alpha,
        )

        self.title("Sign Language Detection")
        self.geometry(f"{WIDTH}x{HEIGHT}")
        self.resizable(False, False)


class VideoCapture:
    def __init__(self, video_source=0):
        self._vid = cv2.VideoCapture(video_source)
        if not self._vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self._vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._vid.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH - 300)

    def get_frame(self) -> Tuple[bool, Optional[cv2.typing.MatLike]]:
        if self._vid.isOpened():
            ret, frame = self._vid.read()
            if ret:
                return (ret, frame)
        return (False, None)

    def __del__(self):
        if self._vid.isOpened():
            self._vid.release()


class MainFrame(tk.Frame):
    def __init__(
        self,
        master: tk.Window,
        predictor: Predictor,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(master, *args, **kwargs)
        self.parent = master
        self.predictor = predictor
        self._c = 0
        self._i = 0
        self._res, self._score = "", 0.0
        self._mem = ""
        self._consecutive = 0
        self._sequence = ""
        self._video_capture = VideoCapture(0)
        self._style = tk.Style()
        self._tts_engine = pyttsx3.init()

        self._render()
        self._update()

    def _render(self):
        side_panel_width = 300
        left_frame = tk.Frame(
            self.parent,
            width=WIDTH - side_panel_width,
            height=HEIGHT,
        )
        left_frame.place(x=0, y=0)

        self.canvas = tk.Canvas(left_frame, width=WIDTH - side_panel_width, height=480)
        self.canvas.place(x=0, y=(HEIGHT - 480) // 2)

        self.sentence_variable = tk.StringVar()
        sentence_label = tk.Label(
            left_frame,
            textvariable=self.sentence_variable,
            width=WIDTH - side_panel_width,
            font=("Open Sans", 24, "bold"),
            anchor="center",
            justify="center",
            bootstyle="primary",
        )
        gap = HEIGHT - ((HEIGHT - 480) // 2 + 480)
        sentence_label.place(
            x=0, y=HEIGHT - gap // 2 - 20, width=WIDTH - side_panel_width
        )

        right_frame = tk.Frame(self.parent, width=side_panel_width, height=HEIGHT)
        right_frame.place(x=WIDTH - side_panel_width, y=0)

        self.score_meter = tk.Meter(
            right_frame,
            bootstyle="primary",
            subtext="Accuracy",
            padding=5,
            metersize=180,
            textfont=("Open Sans", 18),
        )
        self.score_meter.place(x=(side_panel_width - 180) // 2, y=HEIGHT // 2 - 180)

        self.res_textvariable = tk.StringVar(value=" ")
        self.current_pred_label = tk.Label(
            right_frame,
            textvariable=self.res_textvariable,
            font=tk.font.Font(family="Open Sans", size=20, weight="bold"),
            anchor="center",
            justify="center",
        )
        self.current_pred_label.place(
            x=0, y=HEIGHT // 2 + 90 + 20, width=side_panel_width
        )

        self._style.configure("tts_button.TButton", font=("Open Sans", 12))
        self.tts_btn = tk.Button(
            right_frame,
            text="Text to Speech",
            bootstyle="outline",
            style="tts_button.TButton",
            cursor="hand1",
            command=self._tts_btn_clicked,
        )
        self.tts_btn.place(x=20, y=HEIGHT // 2 + 160, width=side_panel_width - 40)

    def _update(self):
        ret, frame = self._video_capture.get_frame()
        if ret:
            img = cv2.flip(frame, 1)
            x1, y1, x2, y2 = 10, 100, 300, 400
            img_cropped = img[y1:y2, x1:x2]

            self._c += 1

            if self._i == 5:
                image_data = cv2.imencode(".jpg", img_cropped)[1].tobytes()
                res_tmp, self._score = self.predictor.predict(image_data)
                self._res = res_tmp
                self._i = 0
                if self._mem == self._res:
                    self._consecutive += 1
                else:
                    self._consecutive = 0
                if self._consecutive == 4 and self._res not in ["nothing"]:
                    if self._res == "space":
                        self._sequence += " "
                    elif self._res == "del":
                        self._sequence = self._sequence[:-1]
                    else:
                        self._sequence += self._res
                    self._consecutive = 0
            self._i += 1
            self.res_textvariable.set(self._res.upper())
            self.sentence_variable.set(self._sequence.upper())
            self.score_meter.configure(amountused=round(self._score * 100, 2))
            self._mem = self._res
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)

            self.photo = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            )
            self.canvas.create_image(40, 0, image=self.photo, anchor=tk.NW)
        self.parent.after(15, self._update)

    def _tts_btn_clicked(self):
        self._tts_engine.say(self._sequence)
        self._tts_engine.runAndWait()


if __name__ == "__main__":
    predictor = Predictor("logs/trained_labels.txt", "logs/trained_graph.pb")
    app = MainApplication()
    MainFrame(app, predictor)
    app.mainloop()
