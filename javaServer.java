// JavaServer.java
import org.deidentifier.arx.*;
import org.deidentifier.arx.criteria.LDiversity;
import py4j.GatewayServer;

public class JavaServer {
    private ARXAnonymizer anonymizer;
    private ARXConfiguration config;

    public JavaServer() {
        anonymizer = new ARXAnonymizer();
        config = ARXConfiguration.create();
    }

    public void addLDiversity(String attribute, double l) {
        config.addPrivacyModel(new LDiversity(attribute, l, false, false));
    }

    public void anonymizeData(Data data) {
        anonymizer.anonymize(data, config);
    }

    public static void main(String[] args) {
        JavaServer app = new JavaServer();
        GatewayServer server = new GatewayServer(app);
        server.start();
        System.out.println("Gateway Server Started");
    }
}