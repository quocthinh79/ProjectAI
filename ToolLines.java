package tool;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;

public class ToolLines {
	public static void main(String[] args) throws IOException {
		String path = "banbe.txt";
		FileInputStream fis = new FileInputStream(path);
		BufferedInputStream bis = new BufferedInputStream(fis);
		InputStreamReader isr = new InputStreamReader(bis, "UTF-8");
		BufferedReader brd = new BufferedReader(isr);
		
		String line;
		String[] lines;
		
		OutputStreamWriter osw = new OutputStreamWriter(new FileOutputStream("bb.txt"),StandardCharsets.UTF_8);
		
		int count = 0;
		while((line = brd.readLine()) != null) {
			lines = split(line);
			String hoi = lines[0].trim();
			String traLoi = lines[1].trim();
			
			String str = ",\r\n\n{\"tag\": \""
					+traLoi+"\",\r\n"
					+ " \"patterns\": [\""+hoi+"\"]"
					+ "}";
			System.out.println(str);
		}
		osw.close();
		brd.close();
	}
	
	public static String[] split(String line) {
		String[] splits = line.split("__eou__");
		return splits;
	}
	
}
