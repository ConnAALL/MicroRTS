package ai.AALL.mcts;

import ai.PassiveAI;
import ai.core.AI;
import ai.jni.Response;
import ai.jni.Responses;
import ai.reward.RewardFunctionInterface;
import rts.units.UnitTypeTable;

/**
 * A vectorized client which lets us run multiple difference environments in
 * parallel (although, on the Java-side of the implementation, there is no
 * actual parallelization and input actions are simply processed sequentially).
 *
 * @author santi and costa
 */
public class JNIGridnetVecClient {
	
	/** Clients of 1 Python/JNI agent vs. 1 Java bot */
    public JNIGridnetClientMCTS[] clients;
    /** Clients of 1 Python/JNI agent vs. 1 Python/JNI agent */
    public JNIGridnetClientSelfPlay[] selfPlayClients;
    /** Clients of 1 Java bot vs. 1 Java bot */
    public JNIBotClient[] botClients;
    
    public int maxSteps;
    public int[] envSteps; 
    public RewardFunctionInterface[] rfs;
    public UnitTypeTable utt;
    boolean partialObs = false;
    public String[] mapPaths;

    // Storage
    
    // [player+environment index][Y][X][
    //		0: do we own a unit without action assignment here?			|-- 1
    //
    //		1: can we do a no-op action?								|
    //		2: can we do a move action?									|
    //		3: can we do a harvest action?								|-- 6 action types
    //		4: can we do a return to base with resources action?		|
    //		5: can we do a produce unit action?							|
    //		6: can we do an attack action								|
    //
    //		7: can we move up/north?									|
    //		8: can we move right/east?									|-- 4 move directions
    //		9: can we move down/south?									|
    //		10: can we move left/west?									|
    //
    //		11: can we harvest up/north?								|
    //		12: can we harvest right/east?								|-- 4 harvest directions
    //		13: can we harvest down/south?								|
    //		14: can we harvest left/west?								|
    //
    //		15: can we return resources to base up/north?				|
    //		16: can we return resources to base right/east?				|-- 4 return resources to base directions
    //		17: can we return resources to base down/south?				|
    //		18: can we return resources to base left/west?				|
    //
    //		19: can we produce unit up/north?							|
    //		20: can we produce unit right/east?							|-- 4 produce unit directions
    //		21: can we produce unit down/south?							|
    //		22: can we produce unit left/west?							|
    //
    //		23: can we produce a unit of type 0?						|
    //		24: can we produce a unit of type 1?						|-- k (= 7) unit types to produce
    //		...: ....													|
    //		29: can we produce a unit of type 6?						|
    //
    //		30: can we attack relative position at ...?					|
    //		31: can we attack relative position at ...?					|-- (maxAttackRange)^2 relative attack locations
    //		...: ...													|
    // ]
    int[][][][] masks;
    
    int[][][][] observation;
    double[][] reward;
    boolean[][] done;
    Response[] rs;
    Responses responses;

    double[] terminalReward1;
    boolean[] terminalDone1;
    double[] terminalReward2;
    boolean[] terminalDone2;

    /**
     * 
     * @param a_num_selfplayenvs Should be a multiple of 2. The number of
     * 	self-play environments (JNI vs. JNI) that will be run is half of this,
     * 	since each environment facilitates two agents.
     * @param a_num_envs The number of environments in which a JNI agent plays
     * 	against a Java bot.
     * @param a_max_steps Maximum duration (in frames) per environment.
     * @param a_rfs Reward functions we want to use to compute rewards at every step.
     * @param a_micrortsPath Path for the microrts root directory (with Java code and maps).
     * @param a_mapPaths Paths (under microrts root dir) for maps to load. The first
     * 	a_num_selfplayenvs entries are used for self-play clients, although only those
     * 	at indices 0, 2, 4, ... are used (indices 1, 3, ... remain unused). The
     * 	next a_num_envs entries are used for clients of JNI agents vs. Java bots.
     * @param a_ai2s Java bots to use in the JNI agent vs. Java bot environments.
     * @param a_utt
     * @param partial_obs
     * @throws Exception
     */
    public JNIGridnetVecClient(int a_num_selfplayenvs, int a_num_envs, int a_max_steps, RewardFunctionInterface[] a_rfs, String a_micrortsPath, String[] a_mapPaths, 
    		AI[] a_ai2s, UnitTypeTable a_utt, boolean partial_obs, int mctsTime) throws Exception {
    	
        maxSteps = a_max_steps;
        utt = a_utt;
        rfs = a_rfs;
        partialObs = partial_obs;
        mapPaths = a_mapPaths;

        // initialize clients
        envSteps = new int[a_num_selfplayenvs + a_num_envs];
        selfPlayClients = new JNIGridnetClientSelfPlay[a_num_selfplayenvs/2];
        for (int i = 0; i < selfPlayClients.length; i++) {
            selfPlayClients[i] = new JNIGridnetClientSelfPlay(a_rfs, a_micrortsPath, mapPaths[i*2], a_utt, partialObs, mctsTime);
        }
        clients = new JNIGridnetClientMCTS[a_num_envs];
        // initialize storage
        for (int i = 0; i < clients.length; i++) {
            assert a_rfs != null : "a_rfs is null";
            assert a_micrortsPath != null : "a_micrortsPath is null";
            assert mapPaths[a_num_selfplayenvs+i] != null : "mapPaths[a_num_selfplayenvs+i] is null";
            assert a_ai2s[i] != null : "a_ai2s[i] is null";
            assert a_utt != null : "a_utt is null";
            clients[i] = new JNIGridnetClientMCTS(a_rfs, a_micrortsPath, mapPaths[a_num_selfplayenvs+i], a_ai2s[i], a_utt, partialObs, mctsTime);
        }
        
        Response r = new JNIGridnetClientMCTS(a_rfs, a_micrortsPath, mapPaths[0], new PassiveAI(a_utt), a_utt, partialObs, mctsTime).reset(0);
        
        int s1 = a_num_selfplayenvs + a_num_envs;
        masks = new int[s1][][][];
        reward = new double[s1][rfs.length];
        done = new boolean[s1][rfs.length];
        terminalReward1 = new double[rfs.length];
        terminalDone1 = new boolean[rfs.length];
        terminalReward2 = new double[rfs.length];
        terminalDone2 = new boolean[rfs.length];
        responses = new Responses(null, null, null);
        rs = new Response[s1];
    }

    /**
     * Constructor for Java-bot-only environments.
     * 
     * @param a_max_steps
     * @param a_rfs
     * @param a_micrortsPath
     * @param a_mapPaths
     * @param a_ai1s
     * @param a_ai2s
     * @param a_utt
     * @param partial_obs
     * @throws Exception
     */
    public JNIGridnetVecClient(int a_max_steps, RewardFunctionInterface[] a_rfs, String a_micrortsPath, String[] a_mapPaths,
        AI[] a_ai1s, AI[] a_ai2s, UnitTypeTable a_utt, boolean partial_obs) throws Exception {
        maxSteps = a_max_steps;
        utt = a_utt;
        rfs = a_rfs;
        partialObs = partial_obs;
        mapPaths = a_mapPaths;

        // initialize clients
        botClients = new JNIBotClient[a_ai2s.length];
        for (int i = 0; i < botClients.length; i++) {
            botClients[i] = new JNIBotClient(a_rfs, a_micrortsPath, mapPaths[i], a_ai1s[i], a_ai2s[i], a_utt, partialObs);
        }
        responses = new Responses(null, null, null);
        rs = new Response[a_ai2s.length];
        reward = new double[a_ai2s.length][rfs.length];
        done = new boolean[a_ai2s.length][rfs.length];
        envSteps = new int[a_ai2s.length];
        terminalReward1 = new double[rfs.length];
        terminalDone1 = new boolean[rfs.length];
    }

    // This reset function takes in the chromosome to evaluate
    public Responses reset(int[] players, float[] chromosome) throws Exception {
        if (botClients != null) {
            for (int i = 0; i < botClients.length; i++) {
                rs[i] = botClients[i].reset(players[i]);
            }
            for (int i = 0; i < rs.length; i++) {
                // observation[i] = rs[i].observation;
                reward[i] = rs[i].reward;
                done[i] = rs[i].done;
            }
            responses.set(null, reward, done);
            return responses;
        }
        
        for (int i = 0; i < selfPlayClients.length; i++) {
            selfPlayClients[i].reset();
            selfPlayClients[i].setMCTSEvalWeights(chromosome);
            rs[i*2] = selfPlayClients[i].getResponse(0);
            rs[i*2+1] = selfPlayClients[i].getResponse(1);
        }
        
        for (int i = selfPlayClients.length * 2; i < players.length; i++) {
            clients[i-selfPlayClients.length*2].setMCTSEvalWeights(chromosome);
            rs[i] = clients[i-selfPlayClients.length*2].reset(players[i]);
        }

        for (int i = 0; i < rs.length; i++) {
            //observation[i] = rs[i].observation;
            reward[i] = rs[i].reward;
            done[i] = rs[i].done;
        }
        
        responses.set(null, reward, done);
        return responses;
    }

    public Responses gameStep(int[] players) throws Exception {
        if (botClients != null) {
            for (int i = 0; i < botClients.length; i++) {
                rs[i] = botClients[i].gameStep(players[i]);
                envSteps[i] += 1;
                if (rs[i].done[0] || envSteps[i] >= maxSteps) {
                    for (int j = 0; j < terminalReward1.length; j++) {
                        terminalReward1[j] = rs[i].reward[j];
                        terminalDone1[j] = rs[i].done[j];
                    }
                    botClients[i].reset(players[i]);
                    for (int j = 0; j < terminalReward1.length; j++) {
                        rs[i].reward[j] = terminalReward1[j];
                        rs[i].done[j] = terminalDone1[j];
                    }
                    rs[i].done[0] = true;
                    envSteps[i] =0;
                }
            }
            for (int i = 0; i < rs.length; i++) {
                // observation[i] = rs[i].observation;
                reward[i] = rs[i].reward;
                done[i] = rs[i].done;
            }
            responses.set(null, reward, done);
            return responses;
        }
        
        for (int i = 0; i < selfPlayClients.length; i++) {
            selfPlayClients[i].gameStep(null, null);
            rs[i*2] = selfPlayClients[i].getResponse(0);
            rs[i*2+1] = selfPlayClients[i].getResponse(1);
            envSteps[i*2] += 1;
            envSteps[i*2+1] += 1;
            if (rs[i*2].done[0] || envSteps[i*2] >= maxSteps) {
                for (int j = 0; j < terminalReward1.length; j++) {
                    terminalReward1[j] = rs[i*2].reward[j];
                    terminalDone1[j] = rs[i*2].done[j];
                    terminalReward2[j] = rs[i*2+1].reward[j];
                    terminalDone2[j] = rs[i*2+1].done[j];
                }

                selfPlayClients[i].reset();
                for (int j = 0; j < terminalReward1.length; j++) {
                    rs[i*2].reward[j] = terminalReward1[j];
                    rs[i*2].done[j] = terminalDone1[j];
                    rs[i*2+1].reward[j] = terminalReward2[j];
                    rs[i*2+1].done[j] = terminalDone2[j];
                }
                rs[i*2].done[0] = true;
                rs[i*2+1].done[0] = true;
                envSteps[i*2] =0;
                envSteps[i*2+1] =0;
            }
        }
        //System.out.printf("selfPlayClients.length %d %n", selfPlayClients.length);
        //System.out.printf("players.length %d %n", players.length);
        for (int i = selfPlayClients.length*2; i < players.length; i++) {
            envSteps[i] += 1;
            //System.out.printf("i-selfPlayClients.length*2 %d %n", i - selfPlayClients.length * 2);
            //System.out.printf("i %d %n", i);
            //System.out.printf("action.length %d %n", action.length);
            rs[i] = clients[i-selfPlayClients.length*2].gameStep(players[i]);
            if (rs[i].done[0] || envSteps[i] >= maxSteps) {
                // TRICKY: note that `clients` already resets the shared `observation`
                // so we need to set the old reward and done to this response
                for (int j = 0; j < rs[i].reward.length; j++) {
                    terminalReward1[j] = rs[i].reward[j];
                    terminalDone1[j] = rs[i].done[j];
                }
                clients[i-selfPlayClients.length*2].reset(players[i]);
                for (int j = 0; j < rs[i].reward.length; j++) {
                    rs[i].reward[j] = terminalReward1[j];
                    rs[i].done[j] = terminalDone1[j];
                }
                rs[i].done[0] = true;
                envSteps[i] = 0;
            }
        }
        
        for (int i = 0; i < rs.length; i++) {
            reward[i] = rs[i].reward;
            done[i] = rs[i].done;
        }
        
        responses.set(null, reward, done);
        return responses;
    }


    /**
     * @param player
     * @return Legal actions masks. For self-play clients, returns masks
     * 	for both players. For the other clients, only returns masks for
     * 	the given player.
     * @throws Exception
     */
    public int[][][][] getMasks(int player) throws Exception {
        for (int i = 0; i < selfPlayClients.length; i++) {
            masks[i*2] = selfPlayClients[i].getMasks(0);
            masks[i*2+1] = selfPlayClients[i].getMasks(1);
        }
        for (int i = selfPlayClients.length*2; i < masks.length; i++) {
            masks[i] = clients[i-selfPlayClients.length*2].getMasks(player);
        }
        return masks;
    }

    public void close() throws Exception {
        if (clients != null) {
            for (JNIGridnetClientMCTS client : clients) {
                client.close();
            }
        }
        if (selfPlayClients != null) {
            for (JNIGridnetClientSelfPlay client : selfPlayClients) {
                client.close();
            }
        }
        if (botClients != null) {
            for (int i = 0; i < botClients.length; i++) {
                botClients[i].close();
            }
        }
    }
}
