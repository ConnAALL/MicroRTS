
/* Generated by Together */

package ai.jneat;

import java.util.*;
import ai.jNeatCommon.*;

/**
 * A Trait is a group of parameters that can be expressed
 * as a group more than one time. Traits save a genetic
 * algorithm from having to search vast parameter landscapes
 * on every node. Instead, each node can simply point to a trait
 * and those traits can evolve on their own
 */
public class Trait extends Neat {
	/**
	 * Numeric identification of trait
	 */
	int trait_id;

	/**
	 * Vector width real value for this object trait
	 */
	double[] params;

	public Trait(Trait t) {
		trait_id = t.trait_id;

		this.params = new double[Neat.p_num_trait_params];
		for (int j = 0; j < Neat.p_num_trait_params; j++)
			params[j] = t.params[j];

	}

	public int getTrait_id() {
		return trait_id;
	}

	public void setTrait_id(int trait_id) {
		this.trait_id = trait_id;
	}

	public double[] getParams() {
		return params;
	}

	public double getParams(int j) {
		return params[j];
	}

	public void setParams(double[] params) {
		this.params = params;
	}

	/**
	 * Insert the method's description here.
	 * Creation date: (11/01/2002 16.40.34)
	 */
	public Trait() {
		trait_id = 0;
		params = new double[Neat.p_num_trait_params];
		for (int j = 0; j < Neat.p_num_trait_params; j++)
			params[j] = 0.0;
	}

	/**
	 * Insert the method's description here.
	 * Creation date: (11/01/2002 16.24.46)
	 * 
	 * @param i int
	 */
	public Trait(String xline) {
		StringTokenizer st;
		String curword;

		st = new StringTokenizer(xline);

		// skip keyword
		curword = st.nextToken();

		// Get the trait_id
		curword = st.nextToken();
		trait_id = Integer.parseInt(curword);
		params = new double[Neat.p_num_trait_params];

		// get real values....
		for (int j = 0; j < Neat.p_num_trait_params; j++) {
			curword = st.nextToken();
			params[j] = Double.parseDouble(curword);

		}

	}

	/**
	 * Insert the method's description here.
	 * Creation date: (14/01/2002 11.46.30)
	 */
	public void viewtext(String header) {
		System.out.print(header);
		// System.out.print("\n +TRAIT: ");
		System.out.print("id = " + trait_id);
		System.out.print(", params = [ ");
		for (int j = 0; j < Neat.p_num_trait_params; j++) {
			System.out.print(params[j] + " ");
		}
		System.out.print("]\n");

	}

	public void op_view() {

		System.out.print(" Trait #" + trait_id + "\t");

		for (int count = 0; count < Neat.p_num_trait_params; count++) {
			System.out.print(params[count] + " ");
		}
		System.out.print("\n");
	}

	/**
	 * Insert the method's description here.
	 * Creation date: (18/01/2002 13.02.06)
	 */

	/**
	 * Special Constructor creates a new Trait which is the average
	 * of 2 existing traits passed in
	 */
	public Trait(Trait t1, Trait t2) {
		params = new double[Neat.p_num_trait_params];
		for (int count = 0; count < Neat.p_num_trait_params; count++)
			params[count] = (t1.params[count] + t2.params[count]) / 2.0;
		trait_id = t1.trait_id;

	}

	/**
	 * Insert the method's description here.
	 * Creation date: (18/01/2002 13.02.06)
	 */
	public void mutate() {
		for (int count = 0; count < Neat.p_num_trait_params; count++) {
			if (NeatRoutine.randfloat() > Neat.p_trait_param_mut_prob) {
				params[count] += (NeatRoutine.randposneg() * NeatRoutine.randfloat()) * Neat.p_trait_mutation_power;
				if (params[count] < 0)
					params[count] = 0;
			}
		}
	}

	/**
	 * Insert the method's description here.
	 * Creation date: (01/02/2002 8.23.59)
	 * 
	 * @param a char
	 */
	public Trait(char a) {
	}

	/**
	 * Insert the method's description here.
	 * Creation date: (01/02/2002 8.23.59)
	 * 
	 * @param a char
	 */

	public Trait(int id, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8,
			double p9) {

		params = new double[Neat.p_num_trait_params];

		trait_id = id;

		params[0] = p1;
		params[1] = p2;
		params[2] = p3;
		params[3] = p4;
		params[4] = p5;
		params[5] = p6;
		params[6] = p7;
		params[7] = 0;
	}

	public void print_to_file(IOseq xFile) {
		StringBuffer s2 = new StringBuffer("");

		s2.append("trait ");
		s2.append(trait_id + " ");

		for (int count = 0; count < Neat.p_num_trait_params; count++) {
			s2.append(params[count] + " ");
		}
		xFile.IOseqWrite(s2.toString());

	}
}