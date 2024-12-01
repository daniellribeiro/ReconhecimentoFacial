package com.projeto.ReconhecimentoFacial;

import org.bytedeco.opencv.opencv_face.EigenFaceRecognizer;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.opencv.core.Core;

@SpringBootApplication
public class ReconhecimentoFacialApplication implements CommandLineRunner {


	public static void main(String[] args) {
		SpringApplication.run(ReconhecimentoFacialApplication.class, args);
	}

	@Override
	public void run(String... args) throws Exception {
		TreinamentoYale.treinamento();
	}
}
