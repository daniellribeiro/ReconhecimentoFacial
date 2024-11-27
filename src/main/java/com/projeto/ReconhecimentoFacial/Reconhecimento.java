package com.projeto.ReconhecimentoFacial;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_face.EigenFaceRecognizer;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.FisherFaceRecognizer;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.opencv.imgproc.Imgproc.COLOR_BGRA2GRAY;

public class Reconhecimento {
    public static void reconhecimento() throws FrameGrabber.Exception {
        System.setProperty("java.awt.headless", "false");

        OpenCVFrameConverter.ToMat converteMat = new OpenCVFrameConverter.ToMat();
        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(1);
        String[] pessoas = {"","Daniel","Ivete"};
        camera.start();
        CanvasFrame cFrame;
        CascadeClassifier detectorFace = new CascadeClassifier("src/main/resources/haarcascade_frontalface_alt.xml");

        //FaceRecognizer reconhecedor = EigenFaceRecognizer.create();
        //reconhecedor.read("src/main/resources/classificadorEigenFaces.yml");
        //reconhecedor.setThreshold(5000);

        //FaceRecognizer reconhecedor = FisherFaceRecognizer.create();
        //reconhecedor.read("src/main/resources/classificadorFisherFaces.yml");
        //reconhecedor.setThreshold(2500);

        FaceRecognizer reconhecedor = LBPHFaceRecognizer.create();
        reconhecedor.read("src/main/resources/classificadorLbph.yml");
        reconhecedor.setThreshold(90);

        Frame frameCapturado = null;
        Mat imagemColorida = new Mat();
        cFrame = new CanvasFrame("Camera");

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

                IntPointer rotulo = new IntPointer(1);
                DoublePointer confianca = new DoublePointer(1);
                reconhecedor.predict(faceCapturada,rotulo,confianca);
                int predicao = rotulo.get(0);
                String nome;
                if (predicao == -1){
                    nome = "Desconhecido";
                }else{
                    nome = pessoas[predicao] + " - " + confianca.get(0);
                }
                int x = Math.max(dadosFace.tl().x() - 10,0);
                int y = Math.max(dadosFace.tl().y() - 10,0);
                putText(imagemColorida, nome, new Point(x,y),FONT_HERSHEY_PLAIN, 1.4, new Scalar(0,255,0,0));
            }
            if (cFrame.isVisible()) {
                cFrame.showImage(converteMat.convert(imagemColorida));
            }else{
                break;
            }
        }
        cFrame.dispose();
        camera.stop();
    }
}

