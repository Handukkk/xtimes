function slide()
{
    var slideVal = document.getElementById('image-slide').value;
    document.getElementById('will-cropped').style.clipPath = "polygon(0 0, " + slideVal + "% 0, " + slideVal + "% 100%, 0 100%" 
    document.getElementById('bar-line').style.left = slideVal + "%"
}