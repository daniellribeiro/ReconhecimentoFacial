package com.projeto.ReconhecimentoFacial;

import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

import java.awt.event.KeyEvent;
import java.util.Scanner;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.opencv.imgproc.Imgproc.COLOR_BGRA2GRAY;

public class Captura {
    public static void captura() throws FrameGrabber.Exception {

        System.out.println("Digite seu id: ");
        Scanner cadastro = new Scanner(System.in);
        int idPessoa = cadastro.nextInt();
        System.out.println("Seu id e " + idPessoa);

        System.setProperty("java.awt.headless", "false");

        OpenCVFrameConverter.ToMat converteMat = new OpenCVFrameConverter.ToMat();
        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(1);
        camera.start();
        CanvasFrame cFrame;
        CascadeClassifier detectorFace = new CascadeClassifier("src/main/resources/haarcascade_frontalface_alt.xml");
        Frame frameCapturado = null;
        org.bytedeco.opencv.opencv_core.Mat imagemColorida = new org.bytedeco.opencv.opencv_core.Mat();
        cFrame = new CanvasFrame("Camera");
        int numeroAmostras = 1000;
        int amostra = 1;

        while (camera.grab() != null) {
            imagemColorida = converteMat.convert(camera.grab());
            Mat imagemCinza = new Mat();
            cvtColor(imagemColorida, imagemCinza, COLOR_BGRA2GRAY);
            RectVector facesDetectadas = new RectVector();
            detectorFace.detectMultiScale(imagemCinza, facesDetectadas, 1.1, 1, 0, new Size(150, 150), new Size(500, 500));

            for (int i = 0; i < facesDetectadas.size(); i++) {
                Rect dadosFace = facesDetectadas.get(i);
                rectangle(imagemColorida, dadosFace, new Scalar(0, 0, 255, 0));
                Mat faceCapturada = new Mat(imagemCinza, dadosFace);
                resize(faceCapturada, faceCapturada, new Size(160, 160));

                if (amostra <= numeroAmostras) {
                    imwrite("src/fotos/pessoa." + idPessoa + "." + amostra + ".jpg", faceCapturada);
                    System.out.println("Foto" + amostra + " capturada \n");
                    amostra++;
                }
            }
            if (cFrame.isVisible()) {
                cFrame.showImage(converteMat.convert(imagemColorida));
            }else{
                break;
            }
        }
        cFrame.dispose();
        camera.stop();
        cadastro.close();
    }
}

