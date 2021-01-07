/*!
    * Start Bootstrap - Grayscale v6.0.3 (https://startbootstrap.com/theme/grayscale)
    * Copyright 2013-2020 Start Bootstrap
    * Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-grayscale/blob/master/LICENSE)
    */
var maxClasses = 10;


function setClassElements() 
{
	var numClasses =  document.getElementById('numClasses').value;
	  console.log(numClasses);
	if(numClasses > maxClasses)
	{
	numClasses = maxClasses;
	document.getElementById('numClasses').value = numClasses

	}

	var i;
	for (i = 1; i <= maxClasses; i++) 
	{
		var classIdName = "class" + i.toString() + "_name";
		var classIdFile = "class" + i.toString()+"_FileUpload";
		
		var tempElement = document.getElementById(classIdName);
		var tempElement_file = document.getElementById(classIdFile);
		if(i<=numClasses)
		{
			tempElement.style.display = 'inline-block';
			tempElement_file.style.display = 'inline-block';
		}
		else
		{
			tempElement.style.display = 'none';
			tempElement_file.style.display = 'none';
		}
	}
}


function trainModel()
{
  	var userName = document.getElementById('userName').value;
  	userName = userName.replace(/\s+/g, '');
  	console.log(userName);
  	var projectName = document.getElementById('projectName').value;  
  	projectName = projectName.replace(/\s+/g, '');
	console.log(projectName);
  
	var numClasses =  document.getElementById('numClasses').value;
	console.log(numClasses);
	
	var formData = new FormData();
	formData.append('userName',userName);
	formData.append('projectName',projectName);
	formData.append('mode','train');
    formData.append('numClasses',numClasses);
    
	
	var uploadSize = 0;
	var i;
  	
  	var classNames = [];
            
            
            
  	for (i = 1; i <= numClasses; i++) 
	{
		var classIdName = "class" + i.toString() + "_name";
		var classIdFile = "class" + i.toString()+"_FileUpload";
		console.log(classIdName);	
		console.log(classIdFile);
		
		var className = document.getElementById(classIdName).value;
		
		if(classNames.includes(className))
		{
			return alert("One or More Class Names are Same. Please enter unique class names.");
		}
		
		classNames.push(className);
		console.log(className);
		
		var filesInput =  document.getElementById(classIdFile).files;
		console.log(filesInput.length);
		
  		if (!filesInput.length)
  		{
    			return alert('Please choose a file to upload first.');
        }
        if (filesInput.length >= 100)
  		{
    			return alert('Please choose less than 100 files per class.');
        }
    		
        for (var j = 0; j < filesInput.length;j++)
		{
			formData.append(className,filesInput[j],filesInput[j].name);
			console.log(className);
			console.log(filesInput[j].name);
			console.log(filesInput[j].size);
			uploadSize = uploadSize + filesInput[j].size/1000
		}
  
  	}
  	
  	console.log("Total Upload Size: ",uploadSize," KBs");
  	if (uploadSize > 9500)
  	{
  		return alert("Total Upload Size: "+ uploadSize +" Kbs. It should be less than 10000 Kbs.");
  		
  	}
    		
	$.ajax({
	      async: true,
	      crossDomain: true,
	      method: 'POST',
	      url: 'https://niwfu8jw24.execute-api.ap-south-1.amazonaws.com/dev/hello',
	      data: formData,
	      processData: false,
	      contentType: false,
	      mimeType: "multipart/form-data",
	})
	.done(function(response){
	  console.log(response);
	  document.getElementById('capstoneResult').textContent = response;
	})
	.fail(function() {alert ("There was an error while sending request."); });
};



function fetchUserInfo()
{
  	var userName = document.getElementById('userName').value;
  	userName = userName.replace(/\s+/g, '');
    console.log('userName');
  	console.log(userName);
  	var projectName = document.getElementById('projectName').value;  
  	projectName = projectName.replace(/\s+/g, '');
	console.log(projectName);
  
	var formData = new FormData();
	formData.append('userName',userName);
	formData.append('projectName',projectName);
	formData.append('mode','userInfo');
        		
	$.ajax({
	      async: true,
	      crossDomain: true,
	      method: 'POST',
	      url: 'https://niwfu8jw24.execute-api.ap-south-1.amazonaws.com/dev/hello',
	      data: formData,
	      processData: false,
	      contentType: false,
	      mimeType: "multipart/form-data",
	})
	.done(function(response){
      var responseObj = JSON.parse(response);
        console.log(responseObj);
        console.log(responseObj.classNames);
        console.log(responseObj.classNames[0]);
        
        if (responseObj.userProjectExists == true)
            {
                document.getElementById('numClasses').value = responseObj.numClasses;
                setClassElements();
                
        
            for (var i=1;i<=responseObj.numClasses;i++)
                {
                    var classIdName = "class" + i.toString() + "_name";
                    console.log(responseObj.classNames[i-1]);
                    document.getElementById(classIdName).value = responseObj.classNames[i-1];  
                    
                }
                
            document.getElementById('inference_container').style.display = 'block';
            
        
	       }
	  
	  
	})
	.fail(function() {alert ("There was an error while sending request."); });
};


function uploadAndInfer(){
    
    
    var userName = document.getElementById('userName').value;
  	userName = userName.replace(/\s+/g, '');
    console.log('userName');
  	console.log(userName);
  	
    var projectName = document.getElementById('projectName').value;  
  	projectName = projectName.replace(/\s+/g, '');
	console.log(projectName);
  
	var formData = new FormData();
	formData.append('userName',userName);
	formData.append('projectName',projectName);
  
    var fileInput = document.getElementById('capstoneInferFileUpload').files;
    if (!fileInput.length){
        return alert('Please choose a file to upload first.');
    }

    var file = fileInput[0];
    var filename = file.name

    formData.append(filename, file);
    console.log(filename);


    $.ajax({
          async: true,
          crossDomain: true,
          method: 'POST',
          url: 'https://23a3jo9e3i.execute-api.ap-south-1.amazonaws.com/dev/infer',
          data: formData,
          processData: false,
          contentType: false,
          mimeType: "multipart/form-data",
    })
    .done(function(response){
      console.log(response);
      jsonResponse = JSON.parse(response);
      document.getElementById('capstoneResult').textContent = jsonResponse.predictedClass;
    })
    .fail(function() {alert ("There was an error while sending Inference request."); });
};



    (function ($) {
    "use strict"; // Start of use strict

    // Smooth scrolling using jQuery easing
    $('a.js-scroll-trigger[href*="#"]:not([href="#"])').click(function () {
        if (
            location.pathname.replace(/^\//, "") ==
                this.pathname.replace(/^\//, "") &&
            location.hostname == this.hostname
        ) {
            var target = $(this.hash);
            target = target.length
                ? target
                : $("[name=" + this.hash.slice(1) + "]");
            if (target.length) {
                $("html, body").animate(
                    {
                        scrollTop: target.offset().top - 70,
                    },
                    1000,
                    "easeInOutExpo"
                );
                return false;
            }
        }
    });

    // Closes responsive menu when a scroll trigger link is clicked
    $(".js-scroll-trigger").click(function () {
        $(".navbar-collapse").collapse("hide");
    });

    // Activate scrollspy to add active class to navbar items on scroll
    $("body").scrollspy({
        target: "#mainNav",
        offset: 100,
    });

    // Collapse Navbar
    var navbarCollapse = function () {
        if ($("#mainNav").offset().top > 100) {
            $("#mainNav").addClass("navbar-shrink");
        } else {
            $("#mainNav").removeClass("navbar-shrink");
        }
    };
    // Collapse now if page is not at top
    navbarCollapse();
    // Collapse the navbar when page is scrolled
    $(window).scroll(navbarCollapse);
})(jQuery); // End of use strict


$('#btnCapstoneFileUpload').click(trainModel);
$('#btnFetchUserInfo').click(fetchUserInfo);
$('#btnCapstoneInference').click(uploadAndInfer);
