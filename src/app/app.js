import P5Wrapper from 'react-p5-wrapper';
import React, {Component} from 'react';
var ReactDOM = require('react-dom');


class Drawing extends Component{


  render(){

    function sketch (p) {
 let rotation = 0;

 p.setup = function () {
   p.createCanvas(600, 400, p.WEBGL);
 };

 p.myCustomRedrawAccordingToNewPropsHandler = function (props) {
   if (props.rotation){
     rotation = props.rotation * Math.PI / 180;
   }
 };

 p.draw = function () {
   p.background(100);
   p.noStroke();
   p.push();
   p.rotateY(rotation);
   p.box(100);
   p.pop();
 };
};

    return(

      <P5Wrapper sketch={sketch} />

    );
  }



};

export default Drawing;
