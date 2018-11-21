import multiprocessing
import os
import time
from datetime import datetime
import psutil

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

window_titles = ['first', 'second', 'third', 'fourth', 'first', 'second', 'third', 'fourth']


def f(uq):
    global emo_end_time
    import cv2
    from imutils.video import WebcamVideoStream
    from emotion_detection import Emotion
    from face_detection import FaceDetection
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    session = tf.Session(config=config)

    # 设置session
    KTF.set_session(session)

    face = FaceDetection()
    emo = Emotion()
    process = multiprocessing.current_process()

    process_pid = psutil.Process(process.pid)
    max_cpu_mem_used, max_cpu_used = -1, -1
    i, url = uq[0], uq[1]
    print("第{}个摄像头的url为:{}".format(i, url))
    vs = WebcamVideoStream(url).start()
    f = open("result.txt", "w")
    f.write(str('time\tpid\tcpu_percent\tcpu_mem_used\tmax_cpu_used\tmax_cpu_mem_used\tface_time(ms)\temo_time(ms)\n'))
    f.close()
    with open("result.txt", "a") as f:
        while True:
            frame = cv2.imread("20181026141713.png")
            # frame = vs.read()
            face_start_time = time.time()
            face_rects = face.predict(frame, single_rect=False)
            face_end_time = time.time() - face_start_time
            if face_rects is not None:
                for face_rect in face_rects:
                    x, y, w, h = face_rect
                    cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                face_frames = face.get_face(frame, face_rects)
                emo_start_time = time.time()
                emotion = emo.predict(face_frames)
                print(emotion)
                emo_end_time = time.time() - emo_start_time
            print("{}:{}".format(i, process.pid))
            cpu_percent = process_pid.cpu_percent()
            cpu_mem_used = getattr(process_pid.memory_info(), "rss")
            max_cpu_used = max(cpu_percent, max_cpu_used)
            max_cpu_mem_used = max(max_cpu_mem_used, cpu_mem_used)
            result = str(datetime.now().strftime(
                "%Y%m%d%H%M%S")) + "\t" + str(process.pid) + "\t" + str(cpu_percent) + "\t" + str(
                cpu_mem_used) + "\t" + str(max_cpu_used) + "\t" + str(max_cpu_mem_used) + "\t" + str(
                face_end_time * 1000) + "\t" + str(emo_end_time * 1000) + "\n"
            # column_name = ['time','pid', 'cpu_percent', 'cpu_mem_used', 'max_cpu_used', 'max_cpu_mem_used']
            f.write(str(result))
            cv2.imshow(window_titles[i], frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        vs.stop()


if __name__ == '__main__':
    cameras = ["rtsp://admin:1234qwer@192.168.16.202/h264/ch1/main/"
               # "rtsp://admin:1234qwer@192.168.16.202/h264/ch1/main/",
               # "rtsp://admin:1234qwer@192.168.16.202/h264/ch1/main/",
               # "rtsp://admin:1234qwer@192.168.16.202/h264/ch1/main/"
               ]

    p = multiprocessing.Pool(processes=1)
    p.map(f, [(i, url) for i, url in enumerate(cameras)])
