import edu.cmu.tetrad.data.*;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.*;
import edu.cmu.tetrad.util.*;

import java.io.*;
import java.util.*;

public class TetradRunner {

    public static void main(String[] args) throws Exception {
        String algorithm = null;
        String dataPath = null;

        // Parse args
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--algorithm") && i + 1 < args.length) {
                algorithm = args[i + 1].toLowerCase();
            }
            if (args[i].equals("--data") && i + 1 < args.length) {
                dataPath = args[i + 1];
            }
        }

        if (algorithm == null || dataPath == null) {
            System.err.println("Usage: java -jar tetrad.jar --algorithm fci --data file.csv");
            System.exit(1);
        }

        // Load dataset
        DataReader reader = new DataReader();
        reader.setDelimiter(DelimiterType.COMMA);
        reader.setHasHeader(true);
        DataSet data = reader.parseTabular(new File(dataPath));

        // Run selected algorithm
        Graph graph = null;
        IndependenceTest test = new IndTestFisherZ(data, 0.05);

        if (algorithm.equals("fci")) {
            Fci fci = new Fci(test);
            fci.setMaxPathLength(-1);
            graph = fci.search();
        } else {
            System.err.println("Unsupported algorithm: " + algorithm);
            System.exit(1);
        }

        // Output result as adjacency list
        for (Edge edge : graph.getEdges()) {
            System.out.println(edge.toString());
        }
    }
}
