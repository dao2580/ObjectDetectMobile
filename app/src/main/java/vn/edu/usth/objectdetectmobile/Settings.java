package vn.edu.usth.objectdetectmobile;

import android.os.Bundle;
import android.view.View;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.SwitchCompat;
import androidx.core.content.ContextCompat;

public class Settings extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.settings);

        // 🔙 Nút quay lại
        ImageButton buttonBack = findViewById(R.id.buttonBack);
        SwitchCompat switchBlur = findViewById(R.id.switchBlur);

        buttonBack.setOnClickListener(v -> finish());

        /*
        // 📥 Các nút download
        ImageView iconDownload1 = findViewById(R.id.iconDownload1);
        ImageView iconDownload2 = findViewById(R.id.iconDownload2);

        // trạng thái hiện tại: true = đang ở chế độ download, false = đang ở chế độ delete
        final boolean[] isDownload = {true};

        iconDownload1.setOnClickListener(v -> {
            if (isDownload[0]) {
                // Đang là download → đổi sang delete
                iconDownload1.setImageResource(R.drawable.ic_delete);
                Toast.makeText(this, "Downloading Depth anything v2...", Toast.LENGTH_SHORT).show();
            } else {
                // Đang là delete → đổi sang download
                iconDownload1.setImageResource(R.drawable.ic_download);
                Toast.makeText(this, "Deleting Depth anything v2...", Toast.LENGTH_SHORT).show();
            }
            // Đảo trạng thái
            isDownload[0] = !isDownload[0];
        });

        iconDownload2.setOnClickListener(v -> {
            if (isDownload[0]) {
                // Đang là download → đổi sang delete
                iconDownload2.setImageResource(R.drawable.ic_delete);
                Toast.makeText(this, "Downloading Depth anything v3...", Toast.LENGTH_SHORT).show();
            } else {
                // Đang là delete → đổi sang download
                iconDownload2.setImageResource(R.drawable.ic_download);
                Toast.makeText(this, "Deleting Depth anything v3...", Toast.LENGTH_SHORT).show();
            }
            // Đảo trạng thái
            isDownload[0] = !isDownload[0];
        });*/

        switchBlur.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) {
                Toast.makeText(this, "Blur Input ON", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "Blur Input OFF", Toast.LENGTH_SHORT).show();
            }
        });
        switchBlur.setThumbTintList(ContextCompat.getColorStateList(this, R.color.switch_thumb_color));
        switchBlur.setTrackTintList(ContextCompat.getColorStateList(this, R.color.switch_track_color));


        /*
        iconDownload1.setOnClickListener(v ->
                Toast.makeText(this, "Downloading Depth anything v2...", Toast.LENGTH_SHORT).show());

        iconDownload2.setOnClickListener(v ->
                Toast.makeText(this, "Downloading Depth anything v3...", Toast.LENGTH_SHORT).show());

        // 🗑️ Các nút delete
        ImageView iconDelete1 = findViewById(R.id.icondelete1);
        ImageView iconDelete2 = findViewById(R.id.icondelete2);

        iconDelete1.setOnClickListener(v ->
                Toast.makeText(this, "Deleting Depth anything v2...", Toast.LENGTH_SHORT).show());

        iconDelete2.setOnClickListener(v ->
                Toast.makeText(this, "Deleting Depth anything v3...", Toast.LENGTH_SHORT).show());*/

        // 📝 Tiêu đề
        TextView titleSettings = findViewById(R.id.titleSettings);
        titleSettings.setText("Settings"); // Có thể thay đổi động nếu cần
    }
}
