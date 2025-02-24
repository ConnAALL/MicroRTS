package ai.AALL;
import ai.jni.JNIInterface;
import rts.PhysicalGameState;

public interface JNIExpertInterface extends JNIInterface{
    public int[] actionMask(PhysicalGameState pgs, final int player);
}
