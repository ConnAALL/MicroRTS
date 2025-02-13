/* Generated by Together */

package ai.jneat;

/* Generated by Together */
import java.util.*;
import java.text.*;
import ai.jNeatCommon.*;

/** Is a genetic codification of gene; */
public class Gene extends Neat {

	/** if a reference to object for identify input/output node and features */
	Link lnk;

	/** is historical marking of node */
	double innovation_num;

	/** how much mutation has changed the link */
	double mutation_num;

	/** is a flag: is TRUE the gene is enabled FALSE otherwise. */
	boolean enable;

	public Link getLnk() {
		return lnk;
	}

	public void setLnk(Link lnk) {
		this.lnk = lnk;
	}

	public double getInnovation_num() {
		return innovation_num;
	}

	public void setInnovation_num(double innovation_num) {
		this.innovation_num = innovation_num;
	}

	public double getMutation_num() {
		return mutation_num;
	}

	public void setMutation_num(double mutation_num) {
		this.mutation_num = mutation_num;
	}

	public boolean getEnable() {
		return enable;
	}

	public void setEnable(boolean enable) {
		this.enable = enable;
	}

	public Gene(Gene g, Trait tp, NNode inode, NNode onode) {
		lnk = new Link(tp, g.lnk.weight, inode, onode, g.lnk.is_recurrent);
		innovation_num = g.innovation_num;
		mutation_num = g.mutation_num;
		enable = g.enable;
	}

	public Gene(String xline, Vector traits, Vector nodes) {

		StringTokenizer st;
		String curword;
		st = new StringTokenizer(xline);
		NNode inode = null;
		NNode onode = null;
		Iterator itr_trait;
		Iterator itr_node;

		// skip keyword 'gene'
		curword = st.nextToken();

		// Get trait_id
		curword = st.nextToken();
		int trait_num = Integer.parseInt(curword);

		// Get input node
		curword = st.nextToken();
		int inode_num = Integer.parseInt(curword);

		// Get output node
		curword = st.nextToken();
		int onode_num = Integer.parseInt(curword);

		// Get weight
		curword = st.nextToken();
		double weight = Double.parseDouble(curword);

		// Get recur
		curword = st.nextToken();
		boolean recur = Integer.parseInt(curword) == 1 ? true : false;

		// Get innovation num
		curword = st.nextToken();
		innovation_num = Double.parseDouble(curword);

		// Get mutation num
		curword = st.nextToken();
		mutation_num = Double.parseDouble(curword);

		// Get enable
		curword = st.nextToken();
		enable = Integer.parseInt(curword) == 1 ? true : false;

		Trait traitptr = null;
		if (trait_num > 0 && traits != null) {

			itr_trait = traits.iterator();
			while (itr_trait.hasNext()) {
				Trait _trait = ((Trait) itr_trait.next());

				if (_trait.trait_id == trait_num) {
					traitptr = _trait;
					break;
				}
			}
		}

		int fnd = 0;

		itr_node = nodes.iterator();
		while (itr_node.hasNext() && fnd < 2) {
			NNode _node = ((NNode) itr_node.next());
			if (_node.node_id == inode_num) {
				inode = _node;
				fnd++;
			}
			if (_node.node_id == onode_num) {
				onode = _node;
				fnd++;
			}

		}
		lnk = new Link(traitptr, weight, inode, onode, recur);

	}

	public void op_view() {

		String mask03 = " 0.000;-0.000";
		DecimalFormat fmt03 = new DecimalFormat(mask03);

		String mask5 = " 0000";
		DecimalFormat fmt5 = new DecimalFormat(mask5);

		System.out.print("\n [Link (" + fmt5.format(lnk.in_node.node_id));
		System.out.print("," + fmt5.format(lnk.out_node.node_id));
		System.out.print("]  innov (" + fmt5.format(innovation_num));

		System.out.print(", mut=" + fmt03.format(mutation_num) + ")");
		System.out.print(" Weight " + fmt03.format(lnk.weight));

		if (lnk.linktrait != null)
			System.out.print(" Link's trait_id " + lnk.linktrait.trait_id);

		if (enable == false)
			System.out.print(" -DISABLED-");

		if (lnk.is_recurrent)
			System.out.print(" -RECUR-");

	}

	public Gene() {
	}

	public Gene(Trait tp, double w, NNode inode, NNode onode, boolean recur, double innov, double mnum) {
		lnk = new Link(tp, w, inode, onode, recur);
		innovation_num = innov;
		mutation_num = mnum;
		enable = true;
	}

	public void print_to_file(IOseq xFile) {

		StringBuffer s2 = new StringBuffer("");

		s2.append("gene ");

		if (lnk.linktrait != null)
			s2.append(" " + lnk.linktrait.trait_id);
		else
			s2.append(" 0");

		s2.append(" " + lnk.in_node.node_id);
		s2.append(" " + lnk.out_node.node_id);
		s2.append(" " + lnk.weight);

		if (lnk.is_recurrent)
			s2.append(" 1");
		else
			s2.append(" 0");

		s2.append(" " + innovation_num);
		s2.append(" " + mutation_num);

		if (enable)
			s2.append(" 1");
		else
			s2.append(" 0");

		xFile.IOseqWrite(s2.toString());

	}
	/**
	 * Insert the method's description here.
	 * Creation date: (24/01/2002 16.59.13)
	 */
}