package com.projeto.ReconhecimentoFacial;

import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_face.*;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;

import static org.bytedeco.opencv.global.opencv_core.CV_32SC1;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class Treinamento {
    public static void treinamento() {
        File diretorio = new File("src/fotos");
        FilenameFilter filtroImagem = new FilenameFilter() {
            @Override
            public boolean accept(File dir, String nome) {
                return nome.endsWith(".jpg");
            }
        };

        File[] arquivos = diretorio.listFiles(filtroImagem);
        MatVector fotos = new MatVector(arquivos.length);
        Mat rotulos = new Mat(arquivos.length,1,CV_32SC1);
        IntBuffer rotulosBuffer = rotulos.createBuffer();
        int contador = 0;
        for (File imagem : arquivos){
            Mat foto = imread(imagem.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
            int classe = Integer.parseInt(imagem.getName().split("\\.")[1]);
            resize(foto,foto,new Size(160,160));
            fotos.put(contador, foto);
            rotulosBuffer.put(contador,classe);
            contador++;
        }
        FaceRecognizer eigenfaces = EigenFaceRecognizer.create();
        FaceRecognizer fisherfaces = FisherFaceRecognizer.create();
        FaceRecognizer lbph = LBPHFaceRecognizer.create();

        eigenfaces.train(fotos,rotulos);
        eigenfaces.save("src/main/resources/classificadorEigenFaces.yml");

        fisherfaces.train(fotos,rotulos);
        fisherfaces.save("src/main/resources/classificadorFisherFaces.yml");

        lbph.train(fotos,rotulos);
        lbph.save("src/main/resources/classificadorLbph.yml");
    }
}
