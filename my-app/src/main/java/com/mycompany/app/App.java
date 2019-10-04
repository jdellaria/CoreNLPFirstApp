package com.mycompany.app;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Properties;

import edu.stanford.nlp.pipeline.*;

import java.util.Properties;
import java.util.stream.Collectors;

//Classifier
import edu.stanford.nlp.ie.AbstractSequenceClassifier;
import edu.stanford.nlp.ie.crf.*;
import edu.stanford.nlp.io.IOUtils;
//import edu.stanford.nlp.ling.CoreLabel;
//import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.sequences.DocumentReaderAndWriter;
import edu.stanford.nlp.util.Triple;
import java.util.List;


/**
 * Hello world!
 *
 */
public class App
{
  /** This is a demo of calling CRFClassifier programmatically.
 *  <p>
 *  Usage: {@code java -mx400m -cp "*" NERDemo [serializedClassifier [fileName]] }
 *  <p>
 *  If arguments aren't specified, they default to
 *  classifiers/english.all.3class.distsim.crf.ser.gz and some hardcoded sample text.
 *  If run with arguments, it shows some of the ways to get k-best labelings and
 *  probabilities out with CRFClassifier. If run without arguments, it shows some of
 *  the alternative output formats that you can get.
 *  <p>
 *  To use CRFClassifier from the command line:
 *  </p><blockquote>
 *  {@code java -mx400m edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier [classifier] -textFile [file] }
 *  </blockquote><p>
 *  Or if the file is already tokenized and one word per line, perhaps in
 *  a tab-separated value format with extra columns for part-of-speech tag,
 *  etc., use the version below (note the 's' instead of the 'x'):
 *  </p><blockquote>
 *  {@code java -mx400m edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier [classifier] -testFile [file] }
 *  </blockquote>
 *
 *  @author Jenny Finkel
 *  @author Christopher Manning
 */
  public static void main(String[] args) throws Exception
  {

//    String serializedClassifier = "classifiers/english.all.3class.distsim.crf.ser.gz";
    String serializedClassifier =""; // = "classifiers/ner-model.ser.gz";


    if (args.length > 0) {
      serializedClassifier = args[0];
    }
    else
    {
      System.out.println("2 arguments are required -> serializedClassifier fileToClassify");
      System.exit(1);
    }

    		    AbstractSequenceClassifier<CoreLabel> classifier = CRFClassifier.getClassifier(serializedClassifier);

    		    /* For either a file to annotate or for the hardcoded text example, this
    		       demo file shows several ways to process the input, for teaching purposes.
    		    */

    		    if (args.length > 1) {

    		      /* For the file, it shows (1) how to run NER on a String, (2) how
    		         to get the entities in the String with character offsets, and
    		         (3) how to run NER on a whole file (without loading it into a String).
    		      */

    		      String fileContents = IOUtils.slurpFile(args[1]);
    		      List<List<CoreLabel>> out = classifier.classify(fileContents);
    		      FileWriter jonWriter = new FileWriter("AnswerAnnotation.txt");
    		      for (List<CoreLabel> sentence : out) {
    		        for (CoreLabel word : sentence) {
    		          String jclassOut = word.word() + '\t' + word.get(CoreAnnotations.AnswerAnnotation.class)+ '\t' + word.get(CoreAnnotations.AnswerProbAnnotation.class) +'\n';
    //		          System.out.print(jclassOut);
    				  jonWriter.append(jclassOut);
    		        }
    //		        System.out.println();
    //				  jonWriter.append("\n");
    		      }
    			  jonWriter.flush();
    			  jonWriter.close();
    		      System.out.println("---");

    		      FileWriter jonWriter2 = new FileWriter("NamedEntityTagProbsAnnotation.txt");
    		      out = classifier.classifyFile(args[1]);
    		      for (List<CoreLabel> sentence : out) {
    		        for (CoreLabel word : sentence) {
    		          String jclassOut = word.word() + '/' + word.get(CoreAnnotations.NamedEntityTagProbsAnnotation.class) + '\n';
    //		          System.out.print(jclassOut);
    				  jonWriter2.append(jclassOut);

    		        }
    //		        System.out.println();
    		      }
    			  jonWriter2.flush();
    			  jonWriter2.close();
    		      System.out.println("---");

    		      FileWriter jonWriter3 = new FileWriter("classifyToCharacterOffsets.txt");
    		      List<Triple<String, Integer, Integer>> list = classifier.classifyToCharacterOffsets(fileContents);
    		      for (Triple<String, Integer, Integer> item : list) {
    //		        System.out.println(item.first() + ": " + fileContents.substring(item.second(), item.third()));
    		        jonWriter3.append(item.first() + ": " + fileContents.substring(item.second(), item.third()));
    		        jonWriter3.append("\n");
    		      }

    			  jonWriter3.flush();
    			  jonWriter3.close();


    		      System.out.println("---");
    		      System.out.println("Ten best entity labelings");
    		      PrintStream o = new PrintStream(new File("TenBestEntityLabelings.txt"));

    			// Store current System.out before assigning a new value
    			  PrintStream console = System.out;
    			// Assign o to output stream
    			  System.setOut(o);

    //		      FileWriter jonWriter3 = new FileWriter("TenBestEntityLabelings.txt");
    		      DocumentReaderAndWriter<CoreLabel> readerAndWriter = classifier.makePlainTextReaderAndWriter();
    		      classifier.classifyAndWriteAnswersKBest(args[1], 10, readerAndWriter);


    		      System.out.println("---");
    		      System.out.println("Per-token marginalized probabilities");
    		      PrintStream o3 = new PrintStream(new File("PerTokenMarginalizedProbabilities.txt"));
    		      System.setOut(o3);
    		      classifier.printProbs(args[1], readerAndWriter);
    		      System.setOut(console);

    		      // -- This code prints out the first order (token pair) clique probabilities.
    		      // -- But that output is a bit overwhelming, so we leave it commented out by default.
    		       System.out.println("---");
    		       System.out.println("First Order Clique Probabilities");
    			   PrintStream o2 = new PrintStream(new File("FirstOrderCliqueProbabilities.txt"));
    		       System.setOut(o2);
    		       ((CRFClassifier) classifier).printFirstOrderProbs(args[1], readerAndWriter);

    			   System.setOut(console);
    		    	System.out.println("Exiting Normally.");
    		    }
  }

  public static void mainNLP(String[] args)
  {
    // set up pipeline properties
    Properties props = new Properties();
    props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner");
    // example customizations (these are commented out but you can uncomment them to see the results

    // disable fine grained ner
    //props.setProperty("ner.applyFineGrained", "false");

    // customize fine grained ner
//    props.setProperty("ner.fine.regexner.mapping", "example.rules");
//    props.setProperty("ner.fine.regexner.ignorecase", "true");

    // add additional rules
//    props.setProperty("ner.additional.regexner.mapping", "example.rules");
//    props.setProperty("ner.additional.regexner.ignorecase", "true");

    // add 2 additional rules files ; set the first one to be case-insensitive
//    props.setProperty("ner.additional.regexner.mapping", "ignorecase=true,example_one.rules;example_two.rules");

    // set up pipeline
    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
    // make an example document
    CoreDocument doc = new CoreDocument("Joe Smith is from Seattle.");
    // annotate the document
    pipeline.annotate(doc);
    // view results
    System.out.println("---");
    System.out.println("entities found");
    for (CoreEntityMention em : doc.entityMentions())
      System.out.println("\tdetected entity: \t"+em.text()+"\t"+em.entityType());
    System.out.println("---");
    System.out.println("tokens and ner tags");
    String tokensAndNERTags = doc.tokens().stream().map(token -> "("+token.word()+","+token.ner()+")").collect(
        Collectors.joining(" "));
    System.out.println(tokensAndNERTags);
  }

    public static void mainOrig( String[] args )
    {
        System.out.println( "Hello World!" );
    }
}
