import P5Wrapper from 'react-p5-wrapper';
import React, {Component} from 'react';
var ReactDOM = require('react-dom');

// tf
import * as tf from '@tensorflow/tfjs';
// tf.js
let x_vals = [];
let y_vals = [];
const loss = (pred, labels) => pred.sub(labels).square().mean();

let a,b;

// optimizer & learning rate
const learningRate = 0.01;
const optimizer = tf.train.sgd(learningRate);


function predict(x){
     const xs = tf.tensor1d(x);
     const ys = xs.mul(a).add(b);
     return ys;
   }

class Drawing extends Component{

  render(){

    function sketch (p) {
       let rotation = 0;

       p.setup = function () {
         p.createCanvas(400, 400);
         p.background(0);
         a = tf.variable(tf.scalar(p.random(1)));
         b = tf.variable(tf.scalar(p.random(1)));

       };

       p.mousePressed = function(){
         let x = p.map(p.mouseX, 0, p.width, 0, 1);
         let y = p.map(p.mouseY, 0, p.height, 1, 0);
         x_vals.push(x);
         y_vals.push(y);
       }



       p.draw = function() {

         tf.tidy(() => {
         if(x_vals.length > 0){
         const ys = tf.tensor1d(y_vals);
         // optimaze
         for (var i=0; i < 20; i++){
         optimizer.minimize(() => loss(predict(x_vals), ys));
           }
         }
       });

         p.background(0);

         p.stroke(350);
         p.strokeWeight(4);

         for (let i = 0; i < x_vals.length; i++){
           let px = p.map(x_vals[i], 0, 1, 0, p.width);
           let py = p.map(y_vals[i], 0, 1, p.height, 0);
           p.point(px, py);
         }


         tf.tidy(() => {

         const xs = [0,1];
         const ys = predict(xs);

         let x1 = p.map(xs[0], 0, 1, 0, p.width);
         let x2 = p.map(xs[1], 0, 1, 0, p.width);

         let lienY = ys.dataSync();
         //console.log(lienY);
         let y1 = p.map(lienY[0], 0, 1, p.height, 0);
         let y2 = p.map(lienY[1], 0, 1, p.height, 0);

         p.line(x1,y1,x2,y2);

});
       }

      };

    return(
      <P5Wrapper sketch={sketch} />
    );
  }
};

export default Drawing;
