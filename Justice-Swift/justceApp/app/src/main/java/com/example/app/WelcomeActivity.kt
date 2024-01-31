package com.example.app

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import org.w3c.dom.Text

class WelcomeActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_welcome)

        val name = intent.getStringExtra(SignInActivity.KEY2)

        val judge = findViewById<Button>(R.id.btnjudge)
        judge.setOnClickListener{
            // new judge activity will open on cliking judge btn
            val openjudgeActivity = Intent(this, judgActivity::class.java)
            startActivity(openjudgeActivity)
        }

       val lawyer = findViewById<Button>(R.id.btnlawyer)
        lawyer.setOnClickListener {
           // new lawyer activity open on clicking lawer btn
            val openlawyerActivity = Intent(this, lawyerActivity::class.java)
            startActivity(openlawyerActivity)
        }

        val customer = findViewById<Button>(R.id.btnconsumer)
        customer.setOnClickListener{
            // new consumer activity open on clicking consumer btn
            val openconsumerActivity = Intent(this, consumerActivity::class.java)
            startActivity(openconsumerActivity)
        }


//        val mail = intent.getStringExtra(SignInActivity.KEY1)
//        val userId = intent.getStringExtra(SignInActivity.KEY3)

        val welcomeText = findViewById<TextView>(R.id.tVWelcome)

//        val mailText = findViewById<TextView>(R.id.tvMail)
//        val idText = findViewById<TextView>(R.id.tvUnique)

        welcomeText.text = "Welcome -> $name"
//        mailText.text = "Mail : $mail"
//        idText.text = "UserId : $userId"

    }
}