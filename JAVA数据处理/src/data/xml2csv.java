package data;
import java.awt.List;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.File;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.LinkedList;
import java.util.Queue;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.nio.*;

public class xml2csv {
	public static int counter;
	public static String savepath="C:\\Users\\54781\\Desktop\\国创\\数据集\\testliu.csv";
	public static String textpath="D:\\en-train\\en";
	public static String testpath="C:\\Users\\54781\\Documents\\Tencent Files\\547818451\\FileRecv\\pan18-author-profiling-test-2018-03-20";
	public static BufferedWriter writer;
	static Hashtable genders = new Hashtable();
	static Hashtable emojiTable = new Hashtable();
	public xml2csv() {}
	public static void initGenders() throws IOException{
		File file = new File(testpath+"\\en.txt");
		FileReader fr = new FileReader(file);
		BufferedReader br = new BufferedReader(fr);
		String line;
		int m=0,f=0;
		while((line = br.readLine())!=null){
			String name = line.substring(0, line.indexOf(":"));
			String gender = line.substring(line.lastIndexOf(":")+1,line.length());
			int gen;
			if(gender.charAt(0)=='f')gen=0;
			else gen=1;
			genders.put(name,gen);
		}
	}
	public static boolean readfile(String filepath) throws FileNotFoundException, IOException {
		try {
			File file = new File(filepath);
			if (!file.isDirectory()){
				String authName=file.getName();
				authName=authName.substring(0,authName.length()-4);
				System.out.println("name=" + authName);
				Pattern p=Pattern.compile("(?<=<document>).*?(?=</document>)");
				FileReader fr= new FileReader(filepath);
				BufferedReader br=new BufferedReader(fr);
				String line =null;
				String q[]=new String[1000];
				int top=1,rear=1;
				while((line=br.readLine())!=null)q[rear++]=line;
				while(top!=rear){
					line = q[top];
					top++;
					if(!line.endsWith(">")){
						q[top]=line+q[top];
						continue;
					}
					int len=line.length();
					for(int i=0;i<len;i++)
						if((int)line.charAt(i)>256||line.charAt(i)=='"'||line.charAt(i)==','){
							StringBuilder sb=new StringBuilder(line);
							sb.setCharAt(i,' ');
							line=sb.toString();
						}
					Matcher m=p.matcher(line);
					while(m.find()){
						String text=m.group();
						text=text.substring(9,text.length()-3);
						writer.append(++counter+","+authName+","+text+","+genders.get(authName)+"\r\n");
						//writer.flush();
						//writer.newLine();
//						writer.println(text);
//						writer.flush();
						//System.out.println(text);
					}
				}
			} 
			else if (file.isDirectory()){
				System.out.println("文件夹");
				String[] filelist = file.list();
				for (int i = 0; i < filelist.length; i++) {
					File readfile = new File(filepath + "\\" + filelist[i]);
					if (!readfile.isDirectory()) {
						readfile(readfile.getAbsolutePath());
					} else if (readfile.isDirectory()) {
						readfile(filepath + "\\" + filelist[i]);
					}
				}

			}

		} catch (FileNotFoundException e) {
			System.out.println("readfile()   Exception:" + e.getMessage());
		}
		return true;
	}
	public static void main(String[] args) {
		try {
			initGenders();
			counter = 0;
			writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(savepath),"UTF-8"));
			readfile(testpath+"\\en\\text");
			writer.flush();
			writer.close();
		} 
		catch (FileNotFoundException ex) {
		} 
		catch (IOException ex) {
		}
		System.out.println("ok");
	}

}