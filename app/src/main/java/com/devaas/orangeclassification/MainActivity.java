/*
 * Copyright (c) 2024 Ashif Ahmed Shuvo
 * LinkedIn: https://linkedin.com/in/me-aas/
 * GitHub: https://github.com/myself-aas
 * Open to Collaborate on innovative projects and research in AI, ML, and related fields.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.devaas.orangeclassification;

import android.os.Build;
import android.os.Bundle;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import com.devaas.orangeclassification.databinding.ActivityMainBinding;

/** Entrypoint for app */
public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        ActivityMainBinding activityMainBinding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(activityMainBinding.getRoot());
    }

    @Override
    public void onBackPressed() {
        // Workaround for potential memory leak in IRequestFinishCallback$Stub on Android Q and above.
        // (https://issuetracker.google.com/issues/139738913)
        // Using finishAfterTransition() instead of super.onBackPressed() to avoid the leak.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            finishAfterTransition();
        } else {
            super.onBackPressed();
        }
    }
}