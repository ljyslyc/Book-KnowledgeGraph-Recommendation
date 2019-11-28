package data;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;

public class getImgPath {
	public static String imgFile="C:\\Users\\54781\\Documents\\Tencent Files\\547818451\\FileRecv\\pan18-author-profiling-test-2018-03-20\\en\\photo";
	public static String savePath="C:\\Users\\54781\\Desktop\\国创\\数据集\\result\\testAuthor.csv";
	public static BufferedWriter writer;
	public static int cnt;
	public getImgPath(){}
	public static boolean readfile(String filepath) throws FileNotFoundException, IOException {
		try {
			File file = new File(filepath);
			String[] filelist = file.list();
			for (int i = 0; i < filelist.length; i++) {
				String author=filelist[i].substring(23);
				writer.append(++cnt+","+author);
				writer.newLine();
			}
			
		}
		catch (FileNotFoundException e) {
			System.out.println("readfile()   Exception:" + e.getMessage());
		}
		return true;
	}
	public static void main(String args[]) throws IOException{
		writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(savePath),"UTF-8"));
		cnt=0;
		readfile(imgFile);
		writer.flush();
		writer.close();
	}
}
